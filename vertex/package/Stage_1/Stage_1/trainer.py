"""Stage-1 training loop implementation."""

from __future__ import annotations

import math
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

from .checkpoints import BestCheckpointSaver
from .data import DataMixer, load_manifest
from .distillation import DistillationConfig, DistillationLoss, TeacherConfig, TeacherModel
from .monitoring import HealthMonitor, MetricAggregator, StructuredLogger
from .models import ModelConfig, Stage1Model, load_stage1_checkpoint
from .utils import WarmupCosineScheduler, config_to_dict, ensure_output_path, get_hf_token
from .utils.attention import _have_flash_attn, pick_attention_backend


@dataclass
class TrainerState:
    step: int = 0


class Stage1Trainer:
    def __init__(self, config, device: Optional[torch.device] = None):
        if not config.resume_gcs_uri:
            raise ValueError("Stage-1 training requires --resume_gcs_uri to point to a checkpoint.")

        self.config = config
        self.device = device or torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dtype = self._select_dtype(config.dtype, self.device)

        if config.block_size > config.seq_len:
            raise ValueError(
                f"block_size ({config.block_size}) must be less than or equal to seq_len ({config.seq_len})"
            )

        self.output_path = config.output_gcs_uri or "./stage1_output"
        if self.output_path and not self.output_path.startswith("gs://"):
            ensure_output_path(self.output_path)

        self.logger = StructuredLogger(self.output_path, run_id=config.run_id, git_sha=self._git_sha())
        self.health = HealthMonitor(self.logger, max_grad_norm=config.max_grad_norm)

        self.attention_backend = "sdpa"
        if self.device.type == "cuda" and self.config.use_flash_attn and _have_flash_attn():
            self.attention_backend = "flash"

        self._tokens_since_log = 0
        self._last_log_time = time.time()
        self.last_tokens_per_sec = 0.0
        self.last_gpu_mem = 0.0
        self.last_grad_norm = 0.0
        self._last_tokens_per_step_effective = 0
        self._last_max_allocated_vram_mb = 0

        project_id = self._detect_project_id()
        secret_name = getattr(config, "hf_secret_name", None)
        hf_token = None
        if secret_name:
            try:
                hf_token = get_hf_token(secret_name, project_id=project_id)
            except ValueError as exc:
                self.logger.health("warning", "hf_secret_project_missing", detail=str(exc))
            except Exception as exc:  # pragma: no cover - secret manager failure
                self.logger.health("warning", "hf_secret_unavailable", detail=str(exc))
        if hf_token:
            for env_key in ("HF_TOKEN", "HF_API_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
                os.environ.setdefault(env_key, hf_token)

        self.tokenizer = self._load_tokenizer(config.tokenizer_name, hf_token)
        vocab_size = len(self.tokenizer)

        self.model_config = ModelConfig(
            vocab_size=vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            max_seq_len=config.seq_len,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
            gradient_checkpointing=config.use_grad_ckpt,
            use_flash_attn=self.attention_backend == "flash",
        )
        self.model = Stage1Model(self.model_config).to(self.device)
        if self.dtype != torch.float32:
            self.model = self.model.to(dtype=self.dtype)

        betas = tuple(float(x) for x in str(config.betas).split(","))
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr, betas=betas, weight_decay=config.weight_decay)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.train_steps,
        )
        self.state = TrainerState()

        if config.best_metric != "val_perplexity" or config.best_metric_mode != "min":
            self.logger.info(
                "checkpoint_metric_override",
                best_metric="val_perplexity",
                mode="min",
            )
            config.best_metric = "val_perplexity"
            config.best_metric_mode = "min"

        distil_cfg = DistillationConfig(
            temperature=config.kd_temperature,
            alpha_start=config.kd_alpha_start,
            alpha_end=config.kd_alpha_end,
            anneal_pct=config.kd_anneal_pct,
            keep_old_logit_l2=config.keep_old_logit_l2,
            keep_old_logit_l2_fade_step=config.keep_old_logit_l2_fade_step,
            keep_old_logit_l2_enable=config.keep_old_logit_l2_enable,
        )
        self.distillation = DistillationLoss(distil_cfg, total_steps=config.train_steps)

        teacher_id = self._resolve_teacher_id()
        self.teacher: Optional[TeacherModel] = None
        if teacher_id or config.teacher_endpoint:
            teacher_cfg = TeacherConfig(
                model_id=teacher_id or "meta-llama/Meta-Llama-3.1-8B",
                endpoint=config.teacher_endpoint,
                hf_secret_name=secret_name,
                hf_token=hf_token,
                hf_cache_dir=getattr(config, "hf_cache_dir", None),
                device=config.device,
                max_batch_size=int(getattr(config, "teacher_max_batch_size", 0)),
            )
            try:
                self.teacher = TeacherModel(teacher_cfg)
            except Exception as exc:  # pragma: no cover - external dependency failures
                self.logger.health("warning", "teacher_unavailable", error=str(exc))
                self.teacher = None

        if not config.dataset_cfg:
            raise ValueError("Stage-1 training requires --dataset_cfg pointing to a manifest JSONL file.")
        manifest = load_manifest(config.dataset_cfg)
        self.mixer = DataMixer(manifest, tool_ratio=config.tool_use_ratio, seed=config.seed)
        self.sample_iterator: Iterator[dict] = self.mixer.iter_samples()
        self.logger.info("dataset_mix", summary=self.mixer.summary())

        self.saver = BestCheckpointSaver(
            self.output_path,
            metric=config.best_metric,
            mode=config.best_metric_mode,
            logger=self.logger,
        )

        self._logged_sources: set[str] = set()

        self._log_startup()
        self._resume_from_checkpoint()

    # ------------------------------------------------------------------
    def _select_dtype(self, dtype: str, device: torch.device) -> torch.dtype:
        if device.type != "cuda":
            return torch.float32
        lowered = str(dtype).lower()
        if lowered in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if lowered in {"float16", "fp16"}:
            return torch.float16
        return torch.float32

    def _load_tokenizer(self, tokenizer_name: str, hf_token: Optional[str]):
        auth_kwargs: Dict[str, object] = {}
        if hf_token:
            auth_kwargs["use_auth_token"] = hf_token
        if getattr(self.config, "hf_cache_dir", None):
            auth_kwargs["cache_dir"] = self.config.hf_cache_dir
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **auth_kwargs)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _git_sha(self) -> Optional[str]:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd()).decode().strip()
        except Exception:  # pragma: no cover - git not always available
            return None

    def _resolve_teacher_id(self) -> Optional[str]:
        name = getattr(self.config, "teacher_name", None)
        if not name:
            return "meta-llama/Meta-Llama-3.1-8B"
        lowered = str(name).lower()
        if lowered in {"none", "null"}:
            return None
        canonical_id = "meta-llama/Meta-Llama-3.1-8B"
        if lowered == canonical_id.lower():
            return canonical_id
        aliases = {
            "meta-llama/meta-llama-3.1-8b": canonical_id,
            "meta-llama/meta-llama-3.1-8b-instruct": canonical_id,
            "llama-3.1-8b": canonical_id,
        }
        if lowered in aliases:
            return aliases[lowered]
        raise ValueError("Stage-1 distillation currently supports only Meta-Llama-3.1-8B as the teacher model")

    def _detect_project_id(self) -> Optional[str]:
        for env_key in (
            "AIP_PROJECT_ID",
            "GOOGLE_CLOUD_PROJECT",
            "PROJECT_ID",
            "CLOUD_ML_PROJECT_ID",
            "GCLOUD_PROJECT",
        ):
            value = os.getenv(env_key)
            if value:
                return value
        return None

    def _log_startup(self) -> None:
        mem = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        self.logger.info(
            "startup",
            device=str(self.device),
            vram=mem,
            seq_len=self.config.seq_len,
            block_size=self.config.block_size,
            d_model=self.model_config.d_model,
            n_layers=self.model_config.n_layers,
            use_flash_attn=self.config.use_flash_attn,
            use_grad_ckpt=self.config.use_grad_ckpt,
            throughput_target=self.config.throughput_tokens,
            grad_accum_steps=self.config.gradient_accumulation_steps,
            attention_backend=self.attention_backend,
        )

    def _resume_from_checkpoint(self) -> None:
        try:
            metadata = load_stage1_checkpoint(self.model, self.config.resume_gcs_uri, device=self.device)
        except Exception as exc:
            self.logger.health("error", "resume_failed", detail=str(exc))
            raise

        self.state.step = metadata.step
        if metadata.optimizer_state:
            try:
                self.optimizer.load_state_dict(metadata.optimizer_state)
            except Exception as exc:  # pragma: no cover - optimizer mismatch
                self.logger.health("warning", "optimizer_state_mismatch", error=str(exc))
        scheduler_state = metadata.scheduler_state or {"step": metadata.step}
        try:
            self.scheduler.load_state_dict(scheduler_state)
        except Exception as exc:  # pragma: no cover
            self.logger.health("warning", "scheduler_state_mismatch", error=str(exc))
        else:
            lrs = list(self.scheduler.get_lr())
            for lr, group in zip(lrs, self.optimizer.param_groups):
                group["lr"] = lr

        self.logger.info(
            "resume_loaded",
            path=self.config.resume_gcs_uri,
            step=metadata.step,
            val_perplexity=metadata.metric_value,
        )

        if self.config.use_checkpoint_saver:
            restored_metric = metadata.metrics.get(self.config.best_metric)
            if restored_metric is None and self.config.best_metric == "val_perplexity":
                restored_metric = metadata.metric_value
            if restored_metric is not None:
                restored_metric = float(restored_metric)
                if math.isnan(restored_metric):
                    self.logger.health(
                        "warning",
                        "resume_metric_nan",
                        detail=(
                            "Best checkpoint metric restored as NaN; subsequent evaluations will overwrite "
                            "the checkpoint once a finite metric is observed."
                        ),
                    )
                else:
                    self.saver.best_value = restored_metric
                    self.logger.info(
                        "checkpoint_best_restored",
                        metric=self.config.best_metric,
                        value=restored_metric,
                    )

    # ------------------------------------------------------------------
    def _encode_sample(self, sample: dict) -> Optional[Dict[str, torch.Tensor]]:
        text = sample.get("text")
        if not text:
            return None
        encoded = self.tokenizer(
            text,
            max_length=self.config.block_size,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _next_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        input_ids = []
        attention_masks = []
        labels = []
        while len(input_ids) < batch_size:
            sample = next(self.sample_iterator)
            encoded = self._encode_sample(sample)
            if encoded is None:
                continue
            source = str(sample.get("source", sample.get("dataset", "unknown")))
            kind = str(sample.get("kind", sample.get("type", "lm")))
            key = f"{source}::{kind}"
            if key not in self._logged_sources:
                self.logger.info("dataset_source", source=source, kind=kind)
                self._logged_sources.add(key)
            input_ids.append(encoded["input_ids"])
            attention_masks.append(encoded["attention_mask"])
            labels.append(encoded["labels"])
        batch = {
            "input_ids": torch.stack(input_ids).to(self.device),
            "attention_mask": torch.stack(attention_masks).to(self.device),
            "labels": torch.stack(labels).to(self.device),
        }
        return batch

    def _eval(self) -> Dict[str, float]:
        self.model.eval()
        aggregator = MetricAggregator()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            batch = self._next_batch(self.config.eval_batch_size)
            with pick_attention_backend(self.config.use_flash_attn):
                logits = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
                teacher_logits = logits.detach()
                if self.teacher is not None:
                    teacher_logits = self.teacher.logits(
                        batch["input_ids"], attention_mask=batch["attention_mask"]
                    )
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["labels"].view(-1),
                ignore_index=-100,
            )
            metrics = self.distillation(self.state.step, logits, teacher_logits, batch["labels"])
            aggregator.update(
                loss=float(loss.item()),
                tokens=int(batch["attention_mask"].sum().item()),
                kd_kl=float(metrics["kd_loss"].item()),
                grad_norm=self.last_grad_norm,
                tokens_per_sec=self.last_tokens_per_sec,
                gpu_mem_reserved=self.last_gpu_mem,
            )
        self.model.train()
        computed = aggregator.compute()
        max_allocated = 0
        if torch.cuda.is_available():
            max_allocated = int(torch.cuda.max_memory_allocated() // 2**20)
        self._last_max_allocated_vram_mb = max_allocated
        computed["tokens_per_step_effective"] = float(self._last_tokens_per_step_effective)
        computed["attn_backend"] = self.attention_backend
        computed["max_allocated_vram_mb"] = float(max_allocated)
        return computed

    # ------------------------------------------------------------------
    def train(self) -> None:
        grad_accum = max(1, self.config.gradient_accumulation_steps)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        start_step = self.state.step
        autocast_enabled = self.device.type == "cuda" and self.dtype in {torch.float16, torch.bfloat16}
        autocast_dtype = self.dtype if autocast_enabled else torch.float32

        for step in range(start_step + 1, self.config.train_steps + 1):
            self.state.step = step
            batch = self._next_batch(self.config.batch_size)
            tokens_in_batch = int(batch["attention_mask"].sum().item())
            self._tokens_since_log += tokens_in_batch
            self._last_tokens_per_step_effective = tokens_in_batch * grad_accum

            try:
                with pick_attention_backend(self.config.use_flash_attn):
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=autocast_dtype,
                        enabled=autocast_enabled,
                    ):
                        logits = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
                        teacher_logits = logits.detach()
                        if self.teacher is not None:
                            teacher_logits = self.teacher.logits(
                                batch["input_ids"], attention_mask=batch["attention_mask"]
                            )
                        distil = self.distillation(
                            step,
                            logits,
                            teacher_logits,
                            batch["labels"],
                        )
                    total_loss = distil["total_loss"]
                    loss = total_loss / grad_accum
                self.health.check_loss(total_loss.detach())
                loss.backward()
            except RuntimeError as err:
                if "out of memory" in str(err).lower():
                    self.health.record_oom()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    if self.config.throughput_tokens and self.config.batch_size > 1:
                        self.config.batch_size = max(1, self.config.batch_size // 2)
                        if self.config.seq_len > 0:
                            grad_accum = max(
                                grad_accum,
                                math.ceil(
                                    self.config.throughput_tokens
                                    / max(1, self.config.batch_size * self.config.seq_len)
                                ),
                            )
                            self.config.gradient_accumulation_steps = grad_accum
                        self.logger.info(
                            "oom_adjustment",
                            batch_size=self.config.batch_size,
                            grad_accum_steps=grad_accum,
                            throughput_tokens=self.config.throughput_tokens,
                        )
                    continue
                raise

            if step % grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.last_grad_norm = float(grad_norm)
                self.health.check_gradients(self.model.parameters())
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            if step % self.config.log_every == 0:
                now = time.time()
                elapsed = now - self._last_log_time
                tokens_per_sec, mem = self.health.report_throughput(self._tokens_since_log, elapsed, step)
                self.last_tokens_per_sec = tokens_per_sec
                self.last_gpu_mem = mem
                self.logger.info(
                    "train_progress",
                    step=step,
                    loss=float(total_loss.detach().item()),
                    lr=float(self.optimizer.param_groups[0]["lr"]),
                    tokens=int(self._tokens_since_log),
                    tokens_per_step_effective=int(self._last_tokens_per_step_effective),
                    attention_backend=self.attention_backend,
                )
                self._last_log_time = now
                self._tokens_since_log = 0

            if step % self.config.eval_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                metrics = self._eval()
                self.logger.metric(step, metrics)
                if self.config.use_checkpoint_saver:
                    self.saver.save(
                        self.model,
                        self.optimizer,
                        self.scheduler.state_dict(),
                        step,
                        metrics,
                        config=config_to_dict(self.config),
                    )


__all__ = ["Stage1Trainer"]
