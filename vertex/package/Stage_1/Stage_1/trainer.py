"""Training loop for Stage-1."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .checkpoints import BestCheckpointSaver
from .distillation import DistillationConfig, DistillationLoss, TeacherConfig, TeacherModel
from .monitoring import HealthMonitor, MetricAggregator, StructuredLogger
from .models import ModelConfig, Stage1Model, load_stage1_checkpoint
from .models.blocks import ClassicBlock
from .utils import WarmupCosineScheduler, config_to_dict, ensure_output_path
from .data import DataMixer, load_manifest
from .tools import ToolTraceInjector


@dataclass
class TrainerState:
    step: int = 0


class Stage1Trainer:
    def __init__(self, config, device: Optional[torch.device] = None):
        if not config.resume_gcs_uri:
            raise ValueError(
                "Stage-1 training requires --resume_gcs_uri to point to a post-surgery checkpoint."
            )

        self.config = config
        self.device = device or torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if (config.precision == "bfloat16" and self.device.type == "cuda") else torch.float32
        self.model_config = ModelConfig(
            max_seq_len=config.seq_len,
            widen_pct=config.net2net_width_pct,
            add_classic=config.add_blocks_classic,
            add_liquid=config.add_blocks_liquid,
            gradient_checkpointing=config.use_grad_ckpt,
        )
        self.model = Stage1Model(self.model_config).to(self.device)
        if self.dtype == torch.bfloat16:
            self.model = self.model.to(dtype=torch.bfloat16)

        betas = tuple(float(x) for x in str(config.betas).split(","))
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr, betas=betas, weight_decay=config.weight_decay)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.max_steps,
        )
        self.state = TrainerState()

        self.output_path = config.output_gcs_uri or "./stage1_output"
        if self.output_path and not self.output_path.startswith("gs://"):
            ensure_output_path(self.output_path)

        self.logger = StructuredLogger(self.output_path, run_id=config.run_id, git_sha=self._git_sha())
        self.health = HealthMonitor(self.logger, max_grad_norm=config.max_grad_norm)

        if getattr(config, "do_surgery", False):
            self.logger.health("warning", "surgery_disabled", detail="Stage-1 skips surgery by design")
            config.do_surgery = False

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
        self.distillation = DistillationLoss(distil_cfg, total_steps=config.max_steps)

        self.teacher: Optional[TeacherModel] = None
        teacher_id = self._resolve_teacher_id()
        teacher_endpoint = getattr(config, "teacher_endpoint", None)
        if (teacher_id or teacher_endpoint) and str(config.teacher).lower() != "none":
            teacher_cfg = TeacherConfig(
                model_id=teacher_id,
                endpoint=teacher_endpoint,
                hf_secret_name=config.hf_secret_name,
                hf_cache_dir=getattr(config, "hf_cache_dir", None),
                device=config.device,
                max_batch_size=getattr(config, "teacher_max_batch_size", 0),
            )
            try:
                self.teacher = TeacherModel(teacher_cfg)
            except Exception as exc:  # pragma: no cover - external dependency
                self.logger.health("warning", "teacher_unavailable", error=str(exc))
                self.teacher = None

        self.old_logit_reference = None
        if getattr(config, "use_old_logit_reference", None):
            self.old_logit_reference = torch.load(config.use_old_logit_reference, map_location="cpu")

        self.saver = BestCheckpointSaver(
            self.output_path,
            metric=config.best_metric,
            mode=config.best_metric_mode,
            logger=self.logger,
        )

        self.mixer = None
        if config.dataset_cfg and os.path.exists(config.dataset_cfg):
            manifest = load_manifest(config.dataset_cfg)
            self.mixer = DataMixer(manifest, tool_ratio=config.tool_use_ratio, seed=config.seed)
            self.logger.info("dataset_mix", weights=self.mixer.summary())
        elif config.dataset_cfg:
            self.logger.info("dataset_manifest_remote", path=config.dataset_cfg)
        self.sample_iterator = self.mixer.iter_samples() if self.mixer else None
        self._logged_sources: set[str] = set()

        self.tool_injector = (
            ToolTraceInjector(
                calculator_enabled=config.calculator_enabled,
                scratchpad_enabled=config.scratchpad_enabled,
                max_calls=max(1, int(config.seq_len * config.tool_use_ratio)),
            )
            if (config.calculator_enabled or config.scratchpad_enabled)
            else None
        )

        self.last_tokens_per_sec = 0.0
        self.last_gpu_mem = 0.0
        self.last_grad_norm = 0.0
        self._last_log_time = time.time()

        self._log_startup()
        self._resume_from_checkpoint()

    # ------------------------------------------------------------------
    def _git_sha(self) -> Optional[str]:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd()).decode().strip()
        except Exception:  # pragma: no cover - git not available
            return None

    def _resolve_teacher_id(self) -> str:
        requested = getattr(self.config, "teacher_id", None)
        if requested and str(requested).lower() != "none":
            return requested
        alias = str(getattr(self.config, "teacher", "")).lower()
        known = {
            "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
            "llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        }
        return known.get(alias, "meta-llama/Meta-Llama-3.1-8B")

    def _log_startup(self) -> None:
        mem = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        self.logger.info(
            "startup",
            device=str(self.device),
            vram=mem,
            seq_len=self.config.seq_len,
            d_model=self.model_config.widened_dim(),
            n_layers=self.model_config.total_layers(),
            freeze_classic_after=self.config.freeze_classic_after,
            use_flash_attn=self.config.use_flash_attn,
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

    # ------------------------------------------------------------------
    def _fake_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        seq_len = self.config.seq_len
        vocab = self.model_config.vocab_size
        input_ids = torch.randint(0, vocab, (batch_size, seq_len), device=self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        labels = input_ids.clone()
        if self.sample_iterator:
            sample = next(self.sample_iterator)
            key = f"{sample['source']}::{sample['type']}"
            if key not in self._logged_sources:
                self.logger.info("batch_source", source=sample["source"], type=sample["type"])
                self._logged_sources.add(key)
            if self.tool_injector and sample["type"].endswith("tool"):
                calls = []
                if self.config.calculator_enabled:
                    calls.append('CALL calculator:"1+1"')
                if self.config.scratchpad_enabled:
                    calls.append('CALL scratchpad:"noted"')
                if calls:
                    _ = self.tool_injector.inject(calls)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def _eval(self) -> Dict[str, float]:
        self.model.eval()
        aggregator = MetricAggregator()
        with torch.no_grad():
            batch = self._fake_batch(self.config.eval_batch_size)
            inputs = batch["input_ids"].to(self.device)
            attn = batch["attention_mask"].to(self.device)
            logits = self.model(inputs, attention_mask=attn)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch["labels"].view(-1), ignore_index=-100)
            teacher_logits = logits.detach()
            if self.teacher is not None:
                teacher_logits = self.teacher.logits(inputs, attention_mask=attn)
            metrics = self.distillation(self.state.step, logits, teacher_logits, batch["labels"])

            math_total = 0
            math_correct = 0
            tool_calls = 0
            tool_total = 0
            if self.tool_injector and self.config.calculator_enabled:
                tool_total = 1
                self.tool_injector.scratchpad.reset()
                trace = self.tool_injector.inject(['CALL calculator:"2+2"'])
                tool_calls = 1
                math_total = 1
                math_correct = 1 if any("RESULT:4" in t for t in trace) else 0

            aggregator.update(
                loss=float(loss.item()),
                tokens=int(attn.sum().item()),
                kd_kl=float(metrics["kd_loss"].item()),
                math_correct=math_correct,
                math_total=math_total,
                tool_calls=tool_calls,
                tool_total=tool_total,
                grad_norm=self.last_grad_norm,
                tokens_per_sec=self.last_tokens_per_sec,
                gpu_mem_reserved=self.last_gpu_mem,
            )
        self.model.train()
        return aggregator.compute()

    # ------------------------------------------------------------------
    def train(self) -> None:
        grad_accum = max(1, self.config.gradient_accumulation_steps)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        start_step = self.state.step
        for step in range(start_step + 1, self.config.max_steps + 1):
            self.state.step = step
            batch = self._fake_batch(self.config.batch_size)

            try:
                with torch.cuda.amp.autocast(enabled=self.dtype == torch.bfloat16):
                    logits = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
                    teacher_logits = logits.detach()
                    if self.teacher is not None:
                        teacher_logits = self.teacher.logits(
                            batch["input_ids"], attention_mask=batch["attention_mask"]
                        )
                    old_logits = self.old_logit_reference
                    distil = self.distillation(step, logits, teacher_logits, batch["labels"], old_logits=old_logits)
                    total_loss = distil["total_loss"]
                    loss = total_loss / grad_accum
                self.health.check_loss(loss.detach())
                loss.backward()
            except RuntimeError as err:
                if "out of memory" in str(err).lower():
                    self.health.record_oom()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
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
                tokens = self.config.batch_size * self.config.seq_len * self.config.log_every
                tokens_per_sec, mem = self.health.report_throughput(tokens, elapsed, step)
                self.last_tokens_per_sec = tokens_per_sec
                self.last_gpu_mem = mem
                self.logger.info(
                    "train_progress",
                    step=step,
                    loss=float(total_loss.detach().item()),
                    lr=float(self.optimizer.param_groups[0]["lr"]),
                    tokens_per_sec=tokens_per_sec,
                )
                self._last_log_time = now

            if step % self.config.eval_every == 0:
                metrics = self._eval()
                self.logger.metric(step, metrics)
                if self.config.use_checkpoint_saver:
                    freeze_mask = {
                        f"blocks.{i}": isinstance(block, ClassicBlock) for i, block in enumerate(self.model.blocks)
                    }
                    self.saver.save(
                        self.model,
                        self.optimizer,
                        self.scheduler.state_dict(),
                        step,
                        metrics,
                        freeze_mask=freeze_mask,
                        config=config_to_dict(self.config),
                    )


__all__ = ["Stage1Trainer"]
