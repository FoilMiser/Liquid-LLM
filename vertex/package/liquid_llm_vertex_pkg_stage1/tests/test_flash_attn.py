import pytest
import torch

from stage1.runtime_setup import enable_flash_attn_if_available


@pytest.mark.cuda
def test_flash_attention_available_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    enabled = enable_flash_attn_if_available()
    if not enabled:
        pytest.skip("FlashAttention kernels unavailable")
    seq = 4
    heads = 2
    hidden = 8
    qkv = torch.randn(1, seq, hidden, device="cuda", dtype=torch.bfloat16)
    attn = torch.nn.MultiheadAttention(embed_dim=hidden, num_heads=heads, batch_first=True, dtype=torch.bfloat16, device="cuda")
    out, _ = attn(qkv, qkv, qkv)
    assert out.shape == (1, seq, hidden)
