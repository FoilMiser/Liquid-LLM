"""Stage-1 model definition and checkpoint loading."""

from __future__ import annotations

import io
import json
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from .blocks import ClassicBlock, LiquidBlock
from .config import ModelConfig
from ..utils.io import open_sharded_file
from torch.utils import checkpoint as checkpoint_utils


class Stage1Model(nn.Module):
    """Transformer language model widened and deepened for Stage-1."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.widened_dim()
        self.token_embed = nn.Embedding(config.vocab_size, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, self.d_model))
        self.dropout = nn.Dropout(config.dropout)
        self.gradient_checkpointing = config.gradient_checkpointing

        blocks = []
        for _ in range(config.n_layers):
            blocks.append(LiquidBlock(self.d_model, config.n_heads, dropout=config.dropout, layer_norm_eps=config.layer_norm_eps))
        for _ in range(config.add_classic):
            blocks.append(ClassicBlock(self.d_model, config.n_heads, dropout=config.dropout, layer_norm_eps=config.layer_norm_eps))
        for _ in range(config.add_liquid):
            blocks.append(LiquidBlock(self.d_model, config.n_heads, dropout=config.dropout, layer_norm_eps=config.layer_norm_eps))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(self.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(self.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        positions = self.pos_embed[:, : input_ids.size(1), :]
        x = self.token_embed(input_ids) + positions
        x = self.dropout(x)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint_utils.checkpoint(lambda inp, blk=block, mask=key_padding_mask: blk(inp, key_padding_mask=mask), x)
            else:
                x = block(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            seq = input_ids
            for _ in range(max_new_tokens):
                logits = self.forward(seq)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                seq = torch.cat([seq, next_token], dim=1)
            return seq

    def write_freeze_mask(self, output_dir: str) -> None:
        mask = {
            f"blocks.{idx}": isinstance(block, ClassicBlock)
            for idx, block in enumerate(self.blocks)
        }
        target = os.path.join(output_dir, "freeze_mask.json") if output_dir else "freeze_mask.json"
        with open_sharded_file(target, "w") as f:
            json.dump(mask, f, indent=2, sort_keys=True)


def load_stage0_state(model: Stage1Model, checkpoint_path: str, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """Load a Stage-0 checkpoint into the widened Stage-1 model."""

    if checkpoint_path.startswith("gs://"):
        with open_sharded_file(checkpoint_path, "rb") as fh:
            buffer = io.BytesIO(fh.read())
        state = torch.load(buffer, map_location=device or "cpu")
    else:
        state = torch.load(checkpoint_path, map_location=device or "cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    stage1_state = model.state_dict()
    remapped: Dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key not in stage1_state:
            continue
        target = stage1_state[key]
        if tensor.shape == target.shape:
            remapped[key] = tensor
        elif tensor.ndim == 2:
            new_tensor = tensor
            pad_rows = target.shape[0] - tensor.shape[0]
            pad_cols = target.shape[1] - tensor.shape[1]
            if pad_rows > 0:
                row_pad = torch.zeros(pad_rows, tensor.shape[1], dtype=tensor.dtype)
                new_tensor = torch.cat([new_tensor, row_pad], dim=0)
            if pad_cols > 0:
                col_pad = torch.zeros(new_tensor.shape[0], pad_cols, dtype=tensor.dtype)
                new_tensor = torch.cat([new_tensor, col_pad], dim=1)
            remapped[key] = new_tensor[: target.shape[0], : target.shape[1]]
        elif tensor.ndim == 1 and tensor.shape[0] != target.shape[0]:
            pad_rows = target.shape[0] - tensor.shape[0]
            if pad_rows <= 0:
                remapped[key] = tensor[: target.shape[0]]
            else:
                pad = torch.zeros(pad_rows, dtype=tensor.dtype)
                remapped[key] = torch.cat([tensor, pad], dim=0)
        else:
            remapped[key] = tensor
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"Warning: missing keys during load: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys ignored: {unexpected}")
    return remapped


__all__ = ["Stage1Model", "load_stage0_state"]
