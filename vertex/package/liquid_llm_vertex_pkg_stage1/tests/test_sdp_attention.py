import sys
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1.model_init import Attention, StudentConfig


def test_attention_uses_sdpa():
    config = StudentConfig(vocab_size=32, hidden_size=16, n_heads=4, n_layers=2, intermediate_size=64, sequence_length=8)
    layer = Attention(config)
    x = torch.randn(2, 8, 16)
    fake_out = torch.zeros(2, config.n_heads, 8, 4)
    with mock.patch("stage1.model_init.F.scaled_dot_product_attention", return_value=fake_out) as sdpa_mock:
        out = layer(x)
    assert sdpa_mock.called
    assert out.shape == (2, 8, 16)
