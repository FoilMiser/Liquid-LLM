"""Model configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32_000
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 15
    max_seq_len: int = 1024
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    gradient_checkpointing: bool = True
    use_flash_attn: bool = True


__all__ = ["ModelConfig"]
