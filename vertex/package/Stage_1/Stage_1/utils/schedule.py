"""Learning rate schedule helpers."""

from __future__ import annotations

import math
from typing import Iterator

import torch


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._step = 0

    def state_dict(self) -> dict:
        return {
            "step": self._step,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._step = state_dict.get("step", 0)

    def get_lr(self) -> Iterator[float]:
        for base_lr in self.base_lrs:
            if self._step < self.warmup_steps:
                yield base_lr * float(self._step + 1) / float(self.warmup_steps)
            else:
                progress = (self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                yield self.min_lr + (base_lr - self.min_lr) * cosine

    def step(self) -> None:
        self._step += 1
        lrs = list(self.get_lr())
        for lr, group in zip(lrs, self.optimizer.param_groups):
            group["lr"] = lr

