"""Minimal DDP utilities."""

from __future__ import annotations

import datetime
import os
from typing import Optional

import torch
import torch.distributed as dist


def setup_ddp(backend: str = "nccl", timeout_seconds: int = 1800) -> Optional[int]:
    if "RANK" not in os.environ:
        return None
    rank = int(os.environ["RANK"])
    dist.init_process_group(
        backend=backend,
        timeout=datetime.timedelta(seconds=timeout_seconds),
    )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return rank


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


__all__ = ["setup_ddp", "cleanup_ddp"]
