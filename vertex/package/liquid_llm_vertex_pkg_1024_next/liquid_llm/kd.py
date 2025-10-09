"""Knowledge distillation helpers."""
from __future__ import annotations

import dataclasses
import json
from typing import Dict, Optional

import numpy as np


@dataclasses.dataclass
class Schedule:
    """Simple schedule supporting constant and linear decay."""

    name: str
    initial: float
    total_steps: int
    final: Optional[float] = None

    def value(self, step: int) -> float:
        if self.final is None or self.total_steps <= 0:
            return self.initial
        step = max(0, min(step, self.total_steps))
        frac = step / float(self.total_steps)
        return self.initial + frac * (self.final - self.initial)

    def to_json(self) -> Dict[str, float]:
        return {
            "name": self.name,
            "initial": self.initial,
            "final": self.final if self.final is not None else self.initial,
            "total_steps": self.total_steps,
        }


def parse_schedule(name: str, base_value: float, spec: Optional[str], total_steps: int) -> Schedule:
    """Parse schedule specification.

    Supported formats:
      - ``None`` -> constant schedule with base value.
      - ``linear:<final>`` -> linear interpolation to ``final`` across ``total_steps``.
    """
    if not spec:
        return Schedule(name=name, initial=base_value, total_steps=total_steps, final=base_value)

    kind, _, payload = spec.partition(":")
    kind = kind.strip().lower()
    if kind == "linear":
        final = float(payload) if payload else base_value
        return Schedule(name=name, initial=base_value, final=final, total_steps=total_steps)
    raise ValueError(f"Unsupported schedule spec: {spec}")


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / max(temperature, 1e-8)
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    probs = np.exp(scaled)
    denom = np.sum(probs, axis=-1, keepdims=True)
    return probs / np.maximum(denom, 1e-12)


def forward_kl(student_logits: np.ndarray, teacher_logits: np.ndarray, temperature: float) -> float:
    """Compute the forward KL divergence (teacher || student)."""
    student = softmax(student_logits, temperature)
    teacher = softmax(teacher_logits, temperature)
    ratio = np.log(np.maximum(teacher, 1e-12)) - np.log(np.maximum(student, 1e-12))
    return float(np.mean(np.sum(teacher * ratio, axis=-1)))


def reverse_kl(student_logits: np.ndarray, teacher_logits: np.ndarray, temperature: float) -> float:
    """Reverse KL (student || teacher) for logging purposes."""
    student = softmax(student_logits, temperature)
    teacher = softmax(teacher_logits, temperature)
    ratio = np.log(np.maximum(student, 1e-12)) - np.log(np.maximum(teacher, 1e-12))
    return float(np.mean(np.sum(student * ratio, axis=-1)))


def entropy_from_logits(logits: np.ndarray, temperature: float) -> float:
    probs = softmax(logits, temperature)
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=-1)
    return float(np.mean(entropy))


def kd_step(student_logits: np.ndarray, teacher_logits: np.ndarray, alpha: float, temperature: float) -> Dict[str, float]:
    """Perform a pseudo KD step returning logging metrics.

    This function does not mutate logits; the caller is expected to update the
    student model externally. Metrics are returned for structured logging.
    """
    kl_fwd = forward_kl(student_logits, teacher_logits, temperature)
    kl_rev = reverse_kl(student_logits, teacher_logits, temperature)
    h_student = entropy_from_logits(student_logits, temperature)
    h_teacher = entropy_from_logits(teacher_logits, temperature)
    kl_scale = temperature ** 2
    loss = alpha * kl_scale * kl_fwd

    return {
        "kl_fwd": kl_fwd,
        "kl_rev": kl_rev,
        "h_student": h_student,
        "h_teacher": h_teacher,
        "kl_scale": kl_scale,
        "loss": loss,
    }


def correlation(student_logits: np.ndarray, teacher_logits: np.ndarray) -> float:
    student = student_logits - np.mean(student_logits)
    teacher = teacher_logits - np.mean(teacher_logits)
    denom = np.sqrt(np.sum(student ** 2)) * np.sqrt(np.sum(teacher ** 2))
    if denom <= 0:
        return 0.0
    return float(np.dot(student, teacher) / denom)


def topk_match(student_logits: np.ndarray, teacher_logits: np.ndarray, k: int) -> float:
    student_top = np.argsort(student_logits)[-k:]
    teacher_top = np.argsort(teacher_logits)[-k:]
    return float(len(set(student_top) & set(teacher_top)) / max(1, k))


def schedule_snapshot(alpha_schedule: Schedule, temp_schedule: Schedule, step: int) -> Dict[str, float]:
    return {
        "step": step,
        "alpha": alpha_schedule.value(step),
        "temperature": temp_schedule.value(step),
    }


def schedule_to_json(alpha_schedule: Schedule, temp_schedule: Schedule) -> str:
    payload = {
        "alpha_schedule": alpha_schedule.to_json(),
        "temp_schedule": temp_schedule.to_json(),
    }
    return json.dumps(payload, sort_keys=True)
