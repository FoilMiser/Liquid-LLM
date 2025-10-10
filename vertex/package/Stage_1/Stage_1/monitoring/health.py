"""Health monitoring utilities."""

from __future__ import annotations

import torch


class HealthMonitor:
    """Performs sanity checks on gradients and runtime state."""

    def __init__(self, logger, max_grad_norm: float = 1.0):
        self.logger = logger
        self.max_grad_norm = max_grad_norm
        self.nan_detected = False
        self.oom_count = 0

    def check_gradients(self, parameters) -> None:
        grads = [p.grad.detach().norm() for p in parameters if p.grad is not None]
        if not grads:
            return
        total_norm = torch.norm(torch.stack(grads))
        if torch.isnan(total_norm) or torch.isinf(total_norm):
            self.nan_detected = True
            self.logger.health("error", "NaN or Inf in gradients", grad_norm=float(total_norm))
            raise FloatingPointError("NaN detected in gradients")
        if total_norm > self.max_grad_norm * 10:
            self.logger.health("error", "Exploding gradients", grad_norm=float(total_norm))
            raise RuntimeError("Exploding gradients")

    def check_loss(self, loss: torch.Tensor) -> None:
        if not torch.isfinite(loss):
            self.nan_detected = True
            self.logger.health("error", "Non-finite loss", loss=float(loss))
            raise FloatingPointError("Non-finite loss")

    def record_oom(self) -> None:
        self.oom_count += 1
        self.logger.health("warning", "Out of memory encountered", oom_count=self.oom_count)
        if self.oom_count >= 3:
            raise RuntimeError("Repeated OOMs detected")

    def report_throughput(self, tokens: int, seconds: float, step: int) -> tuple[float, float]:
        if seconds <= 0:
            return 0.0, 0.0
        tps = tokens / seconds
        mem_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        self.logger.info("throughput", step=step, tokens_per_sec=tps, gpu_mem_reserved=mem_reserved)
        return tps, float(mem_reserved)


__all__ = ["HealthMonitor"]
