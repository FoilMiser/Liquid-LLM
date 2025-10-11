"""Attention backend helpers for Stage-1 models."""

from __future__ import annotations

from contextlib import contextmanager

import torch

try:  # pragma: no cover - optional dependency
    import flash_attn  # noqa: F401

    HAVE_FA = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_FA = False


@contextmanager
def pick_attention_backend(force_fa: bool):
    """Context manager that toggles the preferred attention backend."""

    if not torch.cuda.is_available():
        yield
        return

    try:
        if force_fa and HAVE_FA:
            with torch.backends.cuda.sdp_kernel(  # type: ignore[attr-defined]
                enable_flash=True,
                enable_mem_efficient=True,
                enable_math=False,
            ):
                yield
        else:
            with torch.backends.cuda.sdp_kernel(  # type: ignore[attr-defined]
                enable_flash=False,
                enable_mem_efficient=True,
                enable_math=True,
            ):
                yield
    except Exception:  # pragma: no cover - kernel availability can vary
        yield


__all__ = ["HAVE_FA", "pick_attention_backend"]
