import torch

from stage1.losses import kd_loss


def test_kd_loss_decreases_with_teacher_alignment():
    torch.manual_seed(0)
    student_logits = torch.randn(2, 4, 8, requires_grad=True)
    teacher_logits = torch.randn(2, 4, 8)
    loss1 = kd_loss(student_logits, teacher_logits, temperature=2.0)
    grad, = torch.autograd.grad(loss1, student_logits)
    updated_student = student_logits - 0.1 * grad
    loss2 = kd_loss(updated_student.detach(), teacher_logits, temperature=2.0)
    assert loss2 < loss1
