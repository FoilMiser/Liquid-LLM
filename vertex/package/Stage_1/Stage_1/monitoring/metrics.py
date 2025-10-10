"""Metric aggregation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class MetricState:
    total_loss: float = 0.0
    total_tokens: int = 0
    kd_kl: float = 0.0
    math_correct: int = 0
    math_total: int = 0
    tool_calls: int = 0
    tool_total: int = 0


@dataclass
class MetricAggregator:
    state: MetricState = field(default_factory=MetricState)

    def update(self, loss: float, tokens: int, kd_kl: float | None = None, math_correct: int | None = None, math_total: int | None = None, tool_calls: int | None = None, tool_total: int | None = None) -> None:
        self.state.total_loss += loss * tokens
        self.state.total_tokens += tokens
        if kd_kl is not None:
            self.state.kd_kl = kd_kl
        if math_correct is not None and math_total is not None:
            self.state.math_correct += math_correct
            self.state.math_total += math_total
        if tool_calls is not None and tool_total is not None:
            self.state.tool_calls += tool_calls
            self.state.tool_total += tool_total

    def compute(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self.state.total_tokens > 0:
            metrics["val_perplexity"] = math.exp(self.state.total_loss / self.state.total_tokens)
        metrics["kd_kl"] = self.state.kd_kl
        metrics["math_em"] = self.state.math_correct / max(1, self.state.math_total or 1)
        metrics["tool_call_rate"] = self.state.tool_calls / max(1, self.state.tool_total or 1)
        return metrics


__all__ = ["MetricAggregator"]
