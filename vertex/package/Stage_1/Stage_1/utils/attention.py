"""Attention backend helpers for Stage-1."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch.backends.cuda import sdp_kernel


def _have_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


@contextmanager
def pick_attention_backend(enable_flash: bool) -> Iterator[None]:
    if not torch.cuda.is_available():
        yield
        return
    if enable_flash and _have_flash_attn():
        with sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
            yield
    else:
        with sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True):
            yield


__all__ = ["_have_flash_attn", "pick_attention_backend"]
