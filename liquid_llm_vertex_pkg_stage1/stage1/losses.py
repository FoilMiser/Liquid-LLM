"""Loss functions for Stage 1 training."""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t = temperature
    student_log_probs = F.log_softmax(student_logits / t, dim=-1)
    teacher_probs = F.softmax(teacher_logits / t, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (t ** 2)
    return loss


def ce_loss(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    vocab = student_logits.size(-1)
    loss = F.cross_entropy(student_logits.view(-1, vocab), labels.view(-1), ignore_index=-100)
    return loss


def logit_l2_loss(student_logits: torch.Tensor, baseline_logits: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_logits, baseline_logits)


@dataclass
class LossMixer:
    kd_alpha_start: float = 0.7
    kd_alpha_end: float = 0.4
    kd_anneal_pct: float = 0.3

    def weights(self, step: int, total_steps: int) -> torch.Tensor:
        progress = step / max(1, total_steps)
        if progress > self.kd_anneal_pct:
            frac = 1.0
        else:
            frac = progress / max(self.kd_anneal_pct, 1e-6)
        alpha = self.kd_alpha_start + (self.kd_alpha_end - self.kd_alpha_start) * frac
        beta = 1.0 - alpha
        return torch.tensor([alpha, beta])

    def mix(
        self,
        kd_component: torch.Tensor,
        ce_component: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        alpha, beta = self.weights(step, total_steps)
        return alpha.item() * kd_component + beta.item() * ce_component


def combined_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
    mixer: LossMixer,
    temperature: float,
    baseline_logits: Optional[torch.Tensor] = None,
    baseline_weight: float = 0.0,
    baseline_fade_step: int = 0,
) -> torch.Tensor:
    kd_component = kd_loss(student_logits, teacher_logits, temperature)
    ce_component = ce_loss(student_logits, labels)
    loss = mixer.mix(kd_component, ce_component, step, total_steps)
    if baseline_logits is not None and baseline_weight > 0.0 and step <= baseline_fade_step:
        decay = 1.0 - (step / max(1, baseline_fade_step))
        loss = loss + baseline_weight * decay * logit_l2_loss(student_logits, baseline_logits)
    return loss


__all__ = [
    "kd_loss",
    "ce_loss",
    "logit_l2_loss",
    "LossMixer",
    "combined_loss",
]
