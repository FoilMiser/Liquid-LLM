"""Utility metrics helpers used across training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Meter:
    """Running mean tracker weighted by an optional ``k`` factor."""

    total: float = 0.0
    n: int = 0

    def update(self, value: float, k: int = 1) -> None:
        self.total += float(value) * k
        self.n += k

    def reset(self) -> None:
        self.total = 0.0
        self.n = 0

    @property
    def avg(self) -> float:
        return self.total / max(self.n, 1)


def get_logits(output: Any) -> torch.Tensor:
    """Return raw logits regardless of ``transformers`` output type."""

    if hasattr(output, "logits"):
        return output.logits
    return output


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Token-wise cross entropy using the causal language modelling ignore index."""

    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
    )


def count_active_tokens(labels: torch.Tensor, fallback: int) -> int:
    """
    Count the number of tokens that contribute to loss computation.

    ``fallback`` is typically the full token count for the batch and is used when
    ``labels`` does not contain masked tokens (``-100``) or when no labels are
    provided.
    """

    if labels is None:
        return fallback
    if labels.dtype == torch.long and (labels == -100).any():
        return torch.count_nonzero(labels != -100).item()
    return fallback


def perplexity_from_loss(loss: float) -> float:
    """Compute perplexity from an average cross entropy loss value."""

    return float(torch.exp(torch.tensor(loss))) if loss < float("inf") else float("inf")
