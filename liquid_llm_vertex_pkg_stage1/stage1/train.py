"""Training loop implementation."""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW

from . import gcs_io
from .data import DataModule
from .eval import run_eval
from .losses import LossMixer, combined_loss
from .utils import ThroughputMeter, cosine_with_warmup_schedule

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    import gcsfs


@dataclass
class TeacherProvider:
    mode: str
    logits_dir: str
    teacher_model: Optional[torch.nn.Module] = None

    def _teacher_device(self) -> torch.device:
        if self.teacher_model is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return next(self.teacher_model.parameters()).device

    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.mode == "online":
            if self.teacher_model is None:
                raise RuntimeError("Teacher model not loaded for online mode")
            with torch.no_grad():
                return self.teacher_model(input_ids=input_ids.to(self._teacher_device())).logits
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        batch_logits = []
        for seq in input_ids:
            key = hashlib.sha1(seq.cpu().numpy().tobytes()).hexdigest()
            target_uri = f"{self.logits_dir.rstrip('/')}/{key}.pt"
            local_path = Path("/tmp/teacher_cache") / f"{key}.pt"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                fs.get(target_uri, str(local_path))
                payload = torch.load(local_path, map_location="cpu")
                batch_logits.append(payload["logits"])  # shape [seq, vocab]
            except FileNotFoundError:
                if self.teacher_model is None:
                    raise FileNotFoundError(f"Missing teacher logits shard {target_uri}")
                with torch.no_grad():
                    logits = self.teacher_model(input_ids=seq.unsqueeze(0).to(self._teacher_device())).logits[0].cpu()
                torch.save({"logits": logits}, local_path)
                fs.put(str(local_path), target_uri)
                batch_logits.append(logits)
        return torch.stack(batch_logits, dim=0)


@dataclass
class TrainingConfig:
    learning_rate: float
    betas: tuple
    weight_decay: float
    warmup_steps: int
    max_steps: int
    eval_every: int
    save_every: int
    kd_temperature: float
    kd_alpha_start: float
    kd_alpha_end: float
    kd_anneal_pct: float
    keep_old_logit_l2: float
    keep_old_logit_l2_fade_step: int
    output_gcs_uri: str
    run_id: str
    precision: str = "bfloat16"


def prepare_optimizer(model: torch.nn.Module, config: TrainingConfig) -> AdamW:
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    return optimizer


def train(
    model: torch.nn.Module,
    datamodule: DataModule,
    device: torch.device,
    teacher_provider: TeacherProvider,
    config: TrainingConfig,
    precision_dtype: torch.dtype,
    log_path: Path,
    tensorboard_dir: Optional[Path] = None,
) -> None:
    optimizer = prepare_optimizer(model, config)
    scheduler = cosine_with_warmup_schedule(optimizer, config.warmup_steps, config.max_steps)
    scaler = GradScaler(enabled=(precision_dtype == torch.float16))
    mixer = LossMixer(
        kd_alpha_start=config.kd_alpha_start,
        kd_alpha_end=config.kd_alpha_end,
        kd_anneal_pct=config.kd_anneal_pct,
    )

    tb_writer = None
    if tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))

    train_loader = iter(datamodule.train_dataloader())
    throughput = ThroughputMeter()
    best_perplexity = float("inf")
    step = 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        while step < config.max_steps:
            batch = next(train_loader)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            teacher_logits = teacher_provider.get_logits(input_ids)
            teacher_logits = teacher_logits.to(device)

            optimizer.zero_grad(set_to_none=True)
            start = time.time()
            with torch.cuda.amp.autocast(dtype=precision_dtype, enabled=True):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = combined_loss(
                    logits,
                    teacher_logits,
                    labels,
                    step=step,
                    total_steps=config.max_steps,
                    mixer=mixer,
                    temperature=config.kd_temperature,
                    baseline_logits=None,
                    baseline_weight=config.keep_old_logit_l2,
                    baseline_fade_step=config.keep_old_logit_l2_fade_step,
                )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            duration = time.time() - start
            throughput.update(input_ids.numel(), duration)

            if step % 50 == 0:
                log_line = json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "lr": optimizer.param_groups[0]["lr"],
                        "tokens_per_sec": throughput.tokens_per_second(),
                    }
                )
                log_file.write(log_line + "\n")
                log_file.flush()
                LOGGER.info(log_line)
                gcs_io.upload_logs_periodically(log_path, f"gs://liquid-llm-bucket-2/logs/stage1_console/{config.run_id}.log")
                if tb_writer:
                    tb_writer.add_scalar("train/loss", float(loss.detach().cpu()), step)
                    tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)

            if (step + 1) % config.eval_every == 0:
                metrics = run_eval(model, datamodule, device)
                if tb_writer:
                    for key, value in metrics.items():
                        tb_writer.add_scalar(f"eval/{key}", value, step)
                if metrics["perplexity"] < best_perplexity:
                    best_perplexity = metrics["perplexity"]
                    _save_checkpoint(model, config, suffix="best")

            if (step + 1) % config.save_every == 0:
                _save_checkpoint(model, config, suffix="last")

            step += 1

    if tb_writer:
        tb_writer.close()


def _save_checkpoint(model: torch.nn.Module, config: TrainingConfig, suffix: str) -> None:
    local_path = Path("/tmp") / f"student_{suffix}.pt"
    torch.save(model.state_dict(), local_path)
    if suffix == "best":
        filename = "_best.pt"
    else:
        filename = "last.pt"
    target = f"{config.output_gcs_uri.rstrip('/')}/{config.run_id}/{filename}"
    LOGGER.info("Uploading checkpoint to %s", target)
    gcs_io.gcs_cp(str(local_path), target)


__all__ = ["train", "TrainingConfig", "TeacherProvider"]
