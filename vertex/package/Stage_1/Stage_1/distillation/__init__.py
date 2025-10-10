"""Knowledge distillation utilities."""

from .kd_losses import DistillationConfig, DistillationLoss
from .teacher_llama31_8b import TeacherConfig, TeacherModel

__all__ = ["DistillationLoss", "TeacherModel", "DistillationConfig", "TeacherConfig"]
