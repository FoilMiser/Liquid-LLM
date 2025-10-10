"""Liquid transformer block used for Stage-1."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiquidBlock(nn.Module):
    """A lightweight residual attention block with gating."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, layer_norm_eps: float = 1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        hidden_dim = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask, need_weights=False)
        x = residual + self.dropout(attn_out) * torch.sigmoid(self.gate)
        residual = x
        x = residual + self.dropout(self.ffn(self.norm2(x)))
        return x

