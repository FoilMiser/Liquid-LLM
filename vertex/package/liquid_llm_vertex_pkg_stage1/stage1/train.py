"""Training loop implementation for Stage-1 KD."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from . import losses
from .eval import run_validation
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
        run_id: str,
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
        teacher: Optional[object] = None,
        teacher_mode: str = "online",
        teacher_logits_dir: Optional[str] = None,
        eval_every: int = 1000,
        save_every: int = 2000,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = ensure_local_dir(output_dir)
        self.output_gcs_uri = output_gcs_uri
        self.run_id = run_id
        self.optimizer = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.scheduler = CosineLRSchedule(base_lr=lr, warmup_steps=warmup_steps, total_steps=max_steps)
        self.max_steps = max_steps
        self.kd_temperature = kd_temperature
        self.kd_alpha_schedule = kd_alpha_schedule
        self.ce_beta_schedule = ce_beta_schedule
        self.logit_l2_gamma_schedule = logit_l2_gamma_schedule
        self.logit_reference = logit_reference
        self.scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16" and torch.cuda.is_available()))
        self.precision = precision
        enable_flash_attn_if_available()
        if precision == "bfloat16" and torch.cuda.is_available():
            self.amp_dtype = torch.bfloat16
        elif precision == "fp16":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32
        self.teacher = teacher
        self.teacher_mode = teacher_mode
        self.teacher_logits_dir = teacher_logits_dir
        self.eval_every = eval_every
        self.save_every = save_every
        self._run_uri = f"{self.output_gcs_uri.rstrip('/')}/{self.run_id}" if self.output_gcs_uri else None
        self._metrics_path = Path(self.output_dir) / "val_metrics.jsonl"
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
        if self._run_uri:
            dest = f"{self._run_uri}/{name}"
            local_to_gcs(str(path), dest)
        metadata = RunMetadata(step=step, val_ppl=val_ppl, losses=losses_dict, frozen_blocks=("block_0", "block_1"))
        run_meta_path = Path(self.output_dir) / "run_meta.json"
        with run_meta_path.open("w", encoding="utf-8") as f:
            f.write(metadata.to_json())
        if self._run_uri:
            local_to_gcs(str(run_meta_path), f"{self._run_uri}/run_meta.json")
        logger.info("Saved checkpoint %s", path)

    def _maybe_eval(self, step: int) -> float:
        if self.val_loader is None:
            return float("inf")
        metrics = run_validation(self.model, self.val_loader)
        metrics_payload = {"step": step, **metrics}
        logger.info("Validation metrics at step %d: %s", step, json.dumps(metrics))
        self._write_validation_metrics(metrics_payload)
        return float(metrics.get("perplexity", float("inf")))

    def _write_validation_metrics(self, payload: Dict[str, object]) -> None:
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self._metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        if self._run_uri:
            local_to_gcs(str(self._metrics_path), f"{self._run_uri}/val_metrics.jsonl")

    def _compute_teacher_logits(self, input_ids: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        teacher_logits = batch.get("teacher_logits")
        if teacher_logits is not None:
            return teacher_logits.to(self.device)
        if self.teacher is not None:
            with torch.inference_mode():
                logits = self.teacher.logits(input_ids)
            return logits.to(self.device)
        return None

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
                teacher_logits = self._compute_teacher_logits(input_ids, batch)
                ce = losses.ce_loss(logits, labels)
                kd = logits.new_zeros(())
                if teacher_logits is not None:
                    kd = losses.kd_loss(logits, teacher_logits[:, :-1, :], self.kd_temperature)
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
            if self.eval_every and self.val_loader is not None and step % self.eval_every == 0:
                ppl = self._maybe_eval(step)
                if ppl < best_ppl:
                    best_ppl = ppl
                    self._save_checkpoint("best.pt", step, ppl, metrics)
            if self.save_every and step % self.save_every == 0:
                self._save_checkpoint("last.pt", step, best_ppl, metrics)
            step += 1
        if self._run_uri:
            local_to_gcs(self.output_dir, self._run_uri)

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
