"""Model configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32_000
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 10
    max_seq_len: int = 1024
    dropout: float = 0.0
    widen_pct: float = 10.0
    add_classic: int = 2
    add_liquid: int = 3
    layer_norm_eps: float = 1e-5
    gradient_checkpointing: bool = True
    use_flash_attn: bool = True

    def widened_dim(self) -> int:
        widened = int(round(self.d_model * (1.0 + self.widen_pct / 100.0) / 8.0) * 8)
        widened = max(widened, self.d_model)
        head_multiple = self.n_heads
        if widened % head_multiple != 0:
            widened += head_multiple - (widened % head_multiple)
        return widened

    def total_layers(self) -> int:
        return self.n_layers + self.add_classic + self.add_liquid


__all__ = ["ModelConfig"]
