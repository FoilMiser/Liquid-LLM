"""Best-only checkpoint saver."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch

from ..monitoring.logger import StructuredLogger
from ..utils.io import open_sharded_file


class BestCheckpointSaver:
    def __init__(self, output_path: str, metric: str, mode: str = "min", logger: Optional[StructuredLogger] = None):
        self.output_path = output_path
        self.metric = metric
        self.mode = mode
        self.best_value: Optional[float] = None
        self.logger = logger
        if output_path and not output_path.startswith("gs://"):
            Path(output_path).mkdir(parents=True, exist_ok=True)

    def is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value
        return value > self.best_value

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler_state: dict,
        step: int,
        metrics: dict,
        config: dict | None = None,
    ) -> None:
        value = metrics.get(self.metric)
        if value is None:
            return
        if not self.is_better(value):
            return
        self.best_value = value
        state = {
            "step": step,
            "metrics": metrics,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler_state,
        }
        ckpt_path = os.path.join(self.output_path, "best.pt")
        if ckpt_path.startswith("gs://"):
            with open_sharded_file(ckpt_path, "wb") as fh:
                torch.save(state, fh)
        else:
            torch.save(state, ckpt_path)
        if config is not None:
            config_path = os.path.join(self.output_path, "config_stage1.json")
            with open_sharded_file(config_path, "w") as f:
                json.dump(config, f, indent=2, sort_keys=True)
        if self.logger:
            self.logger.info("checkpoint_saved", metric=self.metric, value=value, step=step)


__all__ = ["BestCheckpointSaver"]
