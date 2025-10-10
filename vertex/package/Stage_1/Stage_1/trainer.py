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
from .models import ModelConfig, Stage1Model, load_stage0_state
from .models.blocks import ClassicBlock
from .utils import WarmupCosineScheduler, config_to_dict
from .data import DataMixer, load_manifest
from .tools import ToolTraceInjector


@dataclass
class TrainerState:
    step: int = 0


class Stage1Trainer:
    def __init__(self, config, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if config.precision == "bfloat16" and torch.cuda.is_available() else torch.float32
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
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr, betas=tuple(float(x) for x in config.betas.split(",")), weight_decay=config.weight_decay)
        self.scheduler = WarmupCosineScheduler(self.optimizer, warmup_steps=config.warmup_steps, total_steps=config.max_steps)
        self.state = TrainerState()
        self.output_path = config.output_gcs_uri or "./stage1_output"
        self.logger = StructuredLogger(self.output_path, run_id=config.run_id, git_sha=self._git_sha())
        self.health = HealthMonitor(self.logger, max_grad_norm=config.max_grad_norm)
        self.distillation = DistillationLoss(
            DistillationConfig(
                temperature=config.kd_temperature,
                alpha_start=config.kd_alpha_start,
                alpha_end=config.kd_alpha_end,
                anneal_pct=config.kd_anneal_pct,
                keep_old_logit_l2=config.keep_old_logit_l2,
                keep_old_logit_l2_fade_step=config.keep_old_logit_l2_fade_step,
                keep_old_logit_l2_enable=config.keep_old_logit_l2_enable,
            ),
            total_steps=config.max_steps,
        )
        self.teacher: Optional[TeacherModel] = None
        if config.teacher and config.teacher.lower() != "none":
            try:
                self.teacher = TeacherModel(TeacherConfig(model_name=config.teacher, hf_secret_name=config.hf_secret_name))
            except Exception as exc:  # pragma: no cover - external dependency
                self.logger.health("warning", "teacher_unavailable", error=str(exc))
                self.teacher = None
        self.old_logit_reference = None
        if config.use_old_logit_reference:
            self.old_logit_reference = torch.load(config.use_old_logit_reference, map_location="cpu")
        self.saver = BestCheckpointSaver(self.output_path, metric=config.best_metric, mode=config.best_metric_mode, logger=self.logger)
        self.mixer = None
        if config.dataset_cfg and os.path.exists(config.dataset_cfg):
            manifest = load_manifest(config.dataset_cfg)
            self.mixer = DataMixer(manifest, tool_ratio=config.tool_use_ratio, seed=config.seed)
            self.logger.info("dataset_mix", weights=self.mixer.summary())
        elif config.dataset_cfg:
            self.logger.info("dataset_manifest_remote", path=config.dataset_cfg)
        self.sample_iterator = self.mixer.iter_samples() if self.mixer else None
        self._logged_sources: set[str] = set()
        self.tool_injector = ToolTraceInjector(
            calculator_enabled=config.calculator_enabled,
            scratchpad_enabled=config.scratchpad_enabled,
            max_calls=max(1, int(config.seq_len * config.tool_use_ratio))
        ) if (config.calculator_enabled or config.scratchpad_enabled) else None
        self.last_tokens_per_sec = 0.0
        self.last_gpu_mem = 0.0
        self.last_grad_norm = 0.0
        self._log_startup()
        if config.resume_gcs_uri:
            try:
                load_stage0_state(self.model, config.resume_gcs_uri, device=self.device)
                self.logger.info("resume_loaded", path=config.resume_gcs_uri)
            except FileNotFoundError:
                self.logger.info("resume_missing", path=config.resume_gcs_uri)
            except Exception as exc:  # pragma: no cover
                self.logger.health("warning", "resume_failed", error=str(exc))

    def _git_sha(self) -> Optional[str]:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd()).decode().strip()
        except Exception:
            return None

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
            teacher_logits = logits
            if self.teacher is not None:
                teacher_logits = self.teacher.logits(inputs, attention_mask=attn)
            metrics = self.distillation(0, logits, teacher_logits, batch["labels"])  # teacher placeholder
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
                loss.item(),
                batch["labels"].numel(),
                kd_kl=metrics["kd_loss"].item(),
                math_correct=math_correct,
                math_total=math_total,
                tool_calls=tool_calls,
                tool_total=tool_total,
            )
        self.model.train()
        return aggregator.compute()

    def train(self) -> None:
        grad_accum = self.config.gradient_accumulation_steps
        self.model.train()
        start = time.time()
        for step in range(1, self.config.max_steps + 1):
            self.state.step = step
            batch = self._fake_batch(self.config.batch_size)
            try:
                with torch.cuda.amp.autocast(enabled=self.dtype == torch.bfloat16):
                    logits = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
                    teacher_logits = logits.detach()
                    if self.teacher is not None:
                        teacher_logits = self.teacher.logits(batch["input_ids"], attention_mask=batch["attention_mask"])
                    old_logits = None
                    if self.old_logit_reference is not None:
                        old_logits = self.old_logit_reference
                    distil = self.distillation(step, logits, teacher_logits, batch["labels"], old_logits=old_logits)
                    loss = distil["total_loss"] / grad_accum
                self.health.check_loss(loss)
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
                elapsed = time.time() - start
                tokens = self.config.batch_size * self.config.seq_len * self.config.log_every
                tps, mem = self.health.report_throughput(tokens, elapsed, step)
                self.last_tokens_per_sec = tps
                self.last_gpu_mem = mem
                start = time.time()
            if step % self.config.eval_every == 0:
                metrics = self._eval()
                metrics.update({
                    "grad_norm": self.last_grad_norm,
                    "tokens_per_sec": self.last_tokens_per_sec,
                    "gpu_mem_reserved": self.last_gpu_mem,
                })
                self.logger.metric(step, metrics)
                if self.config.use_checkpoint_saver:
                    freeze_mask = {f"blocks.{i}": isinstance(block, ClassicBlock) for i, block in enumerate(self.model.blocks)}
                    self.saver.save(self.model, self.optimizer, self.scheduler.state_dict(), step, metrics, freeze_mask=freeze_mask, config=config_to_dict(self.config))
        self.model.write_freeze_mask(self.output_path)


__all__ = ["Stage1Trainer"]
