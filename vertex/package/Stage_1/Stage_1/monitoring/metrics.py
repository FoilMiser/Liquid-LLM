"""Metric aggregation helpers for evaluation and logging."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class MetricState:
    total_loss: float = 0.0
    total_tokens: int = 0
    kd_kl_sum: float = 0.0
    kd_kl_count: int = 0
    math_correct: int = 0
    math_total: int = 0
    tool_calls: int = 0
    tool_total: int = 0
    grad_norm: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    gpu_mem_reserved: Optional[float] = None


@dataclass
class MetricAggregator:
    """Aggregates per-batch metrics into concise evaluation summaries."""

    state: MetricState = field(default_factory=MetricState)

    def update(
        self,
        *,
        loss: float,
        tokens: int,
        kd_kl: float | None = None,
        math_correct: int | None = None,
        math_total: int | None = None,
        tool_calls: int | None = None,
        tool_total: int | None = None,
        grad_norm: float | None = None,
        tokens_per_sec: float | None = None,
        gpu_mem_reserved: float | None = None,
    ) -> None:
        self.state.total_loss += loss * tokens
        self.state.total_tokens += tokens

        if kd_kl is not None:
            self.state.kd_kl_sum += kd_kl
            self.state.kd_kl_count += 1

        if math_correct is not None and math_total is not None:
            self.state.math_correct += math_correct
            self.state.math_total += math_total

        if tool_calls is not None and tool_total is not None:
            self.state.tool_calls += tool_calls
            self.state.tool_total += tool_total

        if grad_norm is not None:
            self.state.grad_norm = grad_norm
        if tokens_per_sec is not None:
            self.state.tokens_per_sec = tokens_per_sec
        if gpu_mem_reserved is not None:
            self.state.gpu_mem_reserved = gpu_mem_reserved

    def compute(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self.state.total_tokens > 0:
            avg_loss = self.state.total_loss / self.state.total_tokens
            metrics["val_perplexity"] = math.exp(avg_loss)
        else:
            metrics["val_perplexity"] = float("inf")

        if self.state.kd_kl_count > 0:
            metrics["kd_kl"] = self.state.kd_kl_sum / self.state.kd_kl_count
        else:
            metrics["kd_kl"] = 0.0

        denom = max(1, self.state.math_total)
        metrics["math_em"] = self.state.math_correct / denom

        tool_denom = max(1, self.state.tool_total)
        metrics["tool_call_rate"] = self.state.tool_calls / tool_denom

        metrics["grad_norm"] = float(self.state.grad_norm or 0.0)
        metrics["tokens_per_sec"] = float(self.state.tokens_per_sec or 0.0)
        metrics["gpu_mem_reserved"] = float(self.state.gpu_mem_reserved or 0.0)

        return metrics


__all__ = ["MetricAggregator"]
