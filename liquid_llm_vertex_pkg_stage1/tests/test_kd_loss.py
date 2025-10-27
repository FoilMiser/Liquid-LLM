import torch

from stage1.losses import kd_loss


def test_kd_temperature_scaling():
    logits_s = torch.randn(2, 3)
    logits_t = torch.randn(2, 3)
    loss_t1 = kd_loss(logits_s, logits_t, temperature=1.0)
    loss_t2 = kd_loss(logits_s, logits_t, temperature=2.0)
    assert loss_t2 >= 0
    assert loss_t1 >= 0
