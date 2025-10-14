"""Utility helpers for Stage 1 training."""

import argparse
import dataclasses
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    LOGGER.info("Seed set to %d", seed)


@dataclass
class ThroughputMeter:
    window: int = 50
    times: Tuple[float, ...] = dataclasses.field(default_factory=tuple)

    def update(self, n_tokens: int, duration_s: float) -> None:
        entry = (time.time(), n_tokens, duration_s)
        self.times = (self.times + (entry,))[-self.window :]

    def tokens_per_second(self) -> float:
        if not self.times:
            return 0.0
        total_tokens = sum(t[1] for t in self.times)
        total_time = sum(t[2] for t in self.times)
        return total_tokens / max(total_time, 1e-6)


def cosine_with_warmup_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in {"yes", "true", "t", "1", "y"}:
        return True
    if v.lower() in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_betas(betas: str) -> Tuple[float, float]:
    parts = betas.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Betas must be 'beta1,beta2'.")
    return float(parts[0]), float(parts[1])


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def chunked(iterable: Iterable, n: int) -> Iterator[Tuple]:
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == n:
            yield tuple(chunk)
            chunk = []
    if chunk:
        yield tuple(chunk)


__all__ = [
    "set_seed",
    "ThroughputMeter",
    "cosine_with_warmup_schedule",
    "str2bool",
    "parse_betas",
    "ensure_dir",
    "chunked",
]
