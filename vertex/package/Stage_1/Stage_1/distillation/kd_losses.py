"""Knowledge distillation losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DistillationConfig:
    temperature: float = 2.0
    alpha_start: float = 0.7
    alpha_end: float = 0.4
    anneal_pct: float = 0.3
    keep_old_logit_l2: float = 0.1
    keep_old_logit_l2_fade_step: int = 30_000
    keep_old_logit_l2_enable: bool = True


class DistillationLoss:
    def __init__(self, config: DistillationConfig, total_steps: int):
        self.cfg = config
        self.total_steps = total_steps

    def kd_alpha(self, step: int) -> float:
        anneal_steps = max(1, int(self.cfg.anneal_pct * self.total_steps))
        if step >= anneal_steps:
            return self.cfg.alpha_end
        ratio = step / float(anneal_steps)
        return self.cfg.alpha_start + ratio * (self.cfg.alpha_end - self.cfg.alpha_start)

    def __call__(self, step: int, student_logits: torch.Tensor, teacher_logits: torch.Tensor, target_ids: torch.Tensor, old_logits: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        temperature = self.cfg.temperature
        kd_alpha = self.kd_alpha(step)
        kd_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        kd_loss = F.kl_div(kd_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

        ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), target_ids.view(-1), ignore_index=-100)
        total_loss = kd_alpha * kd_loss + (1 - kd_alpha) * ce_loss

        l2_loss = torch.tensor(0.0, device=student_logits.device)
        if self.cfg.keep_old_logit_l2_enable and old_logits is not None and self.cfg.keep_old_logit_l2 > 0:
            if step <= self.cfg.keep_old_logit_l2_fade_step:
                weight = self.cfg.keep_old_logit_l2 * max(0.0, 1.0 - step / max(1, self.cfg.keep_old_logit_l2_fade_step))
                l2_loss = F.mse_loss(student_logits, old_logits) * weight
                total_loss = total_loss + l2_loss

        return {
            "total_loss": total_loss,
            "kd_loss": kd_loss.detach(),
            "ce_loss": ce_loss.detach(),
            "kd_alpha": torch.tensor(kd_alpha, device=student_logits.device),
            "l2_loss": l2_loss.detach(),
        }


__all__ = ["DistillationLoss", "DistillationConfig"]
