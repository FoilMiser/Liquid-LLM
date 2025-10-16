"""Training loop implementation for Stage-1 KD."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from . import losses
from .eval import evaluate_perplexity
from .gcs_io import ensure_local_dir, local_to_gcs
from .runtime_setup import enable_flash_attn_if_available
from .utils import AnnealingSchedule, CosineLRSchedule, RunMetadata, configure_logging, json_log

logger = configure_logging()


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: torch.device,
        output_dir: str,
        output_gcs_uri: Optional[str],
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
        warmup_steps: int,
        max_steps: int,
        kd_temperature: float,
        kd_alpha_schedule: AnnealingSchedule,
        ce_beta_schedule: AnnealingSchedule,
        logit_l2_gamma_schedule: AnnealingSchedule,
        logit_reference: Optional[torch.Tensor] = None,
        precision: str = "bfloat16",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = ensure_local_dir(output_dir)
        self.output_gcs_uri = output_gcs_uri
        self.optimizer = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.scheduler = CosineLRSchedule(base_lr=lr, warmup_steps=warmup_steps, total_steps=max_steps)
        self.max_steps = max_steps
        self.kd_temperature = kd_temperature
        self.kd_alpha_schedule = kd_alpha_schedule
        self.ce_beta_schedule = ce_beta_schedule
        self.logit_l2_gamma_schedule = logit_l2_gamma_schedule
        self.logit_reference = logit_reference
        self.scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))
        self.precision = precision
        enable_flash_attn_if_available()
        if precision == "bfloat16" and torch.cuda.is_available():
            self.amp_dtype = torch.bfloat16
        elif precision == "fp16":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self._provenance_synced = False

    def _save_checkpoint(self, name: str, step: int, val_ppl: float, losses_dict: Dict[str, float]) -> None:
        self._ensure_provenance_artifacts()
        path = Path(self.output_dir) / name
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
        }
        torch.save(state, path)
        metadata = RunMetadata(step=step, val_ppl=val_ppl, losses=losses_dict, frozen_blocks=("block_0", "block_1"))
        with (Path(self.output_dir) / "run_meta.json").open("w", encoding="utf-8") as f:
            f.write(metadata.to_json())
        logger.info("Saved checkpoint %s", path)

    def _maybe_eval(self, step: int) -> float:
        if self.val_loader is None:
            return float("inf")
        ppl = evaluate_perplexity(self.model, self.val_loader)
        logger.info("Validation perplexity at step %d: %.4f", step, ppl)
        return ppl

    def train(self) -> None:
        best_ppl = float("inf")
        step = 0
        data_iter = iter(self.train_loader)
        use_autocast = self.device.type != "cpu"
        while step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
            input_ids = batch["input_ids"].to(self.device)
            labels = input_ids[:, 1:].contiguous()
            student_inputs = input_ids[:, :-1]
            autocast_ctx = torch.autocast(device_type=self.device.type, dtype=self.amp_dtype) if use_autocast else torch.cuda.amp.autocast(enabled=False)
            with autocast_ctx:
                logits = self.model(student_inputs)
                logits = logits[:, :-1]
                teacher_logits = batch.get("teacher_logits")
                if teacher_logits is not None:
                    teacher_logits = teacher_logits.to(self.device)
                ce = losses.ce_loss(logits, labels)
                kd = (
                    losses.kd_loss(logits, teacher_logits, self.kd_temperature)
                    if teacher_logits is not None
                    else torch.tensor(0.0, device=self.device)
                )
                l2 = losses.logit_l2(logits, self.logit_reference)
                alpha = self.kd_alpha_schedule.value(step, self.max_steps)
                beta = self.ce_beta_schedule.value(step, self.max_steps)
                gamma = self.logit_l2_gamma_schedule.value(step, self.max_steps)
                total_loss = alpha * kd + beta * ce + gamma * l2
            self.optimizer.zero_grad()
            if self.scaler.is_enabled():
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            lr = self.scheduler.value(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            metrics = {
                "step": step,
                "loss_total": float(total_loss.detach().cpu()),
                "loss_ce": float(ce.detach().cpu()),
                "loss_kd": float(kd.detach().cpu()),
                "loss_l2": float(l2.detach().cpu()),
                "lr": lr,
            }
            json_log(logger, metrics)
            if step % 1000 == 0 and self.val_loader is not None:
                ppl = self._maybe_eval(step)
                if ppl < best_ppl:
                    best_ppl = ppl
                    self._save_checkpoint("best.pt", step, ppl, metrics)
            if step % 2000 == 0:
                self._save_checkpoint("last.pt", step, best_ppl, metrics)
            step += 1
        if self.output_gcs_uri:
            local_to_gcs(self.output_dir, self.output_gcs_uri)

    def _ensure_provenance_artifacts(self) -> None:
        if self._provenance_synced:
            return
        source_dir = Path(os.environ.get("STAGE1_DATA_PROVENANCE_DIR", self.output_dir))
        target_dir = Path(self.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in ("data_readiness.json", "datasets_yaml_snapshot.yaml", "manifest_snapshot.jsonl"):
            src = source_dir / name
            dst = target_dir / name
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)
        self._provenance_synced = True
