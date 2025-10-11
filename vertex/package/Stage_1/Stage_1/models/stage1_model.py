"""Stage-1 model definition and checkpoint utilities."""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils import checkpoint as checkpoint_utils

from ..utils.io import open_sharded_file
from .blocks import LiquidBlock
from .config import ModelConfig
from .attention import pick_attention_backend


@dataclass
class CheckpointMetadata:
    """Metadata parsed from a Stage-1 checkpoint."""

    step: int
    metric_value: Optional[float]
    metrics: Dict[str, float]
    optimizer_state: Optional[dict] = None
    scheduler_state: Optional[dict] = None


class Stage1Model(nn.Module):
    """Transformer language model used for Stage-1."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.token_embed = nn.Embedding(config.vocab_size, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, self.d_model))
        self.dropout = nn.Dropout(config.dropout)
        self.gradient_checkpointing = bool(config.gradient_checkpointing)

        blocks: list[nn.Module] = []
        blocks.extend(
            LiquidBlock(
                self.d_model,
                config.n_heads,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.n_layers)
        )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(self.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(self.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        positions = self.pos_embed[:, : input_ids.size(1), :]
        hidden_states = self.token_embed(input_ids) + positions
        hidden_states = self.dropout(hidden_states)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:

                def _forward_block(tensor, blk=block, mask=key_padding_mask):
                    with pick_attention_backend(self.config.use_flash_attn):
                        return blk(tensor, key_padding_mask=mask)

                hidden_states = checkpoint_utils.checkpoint(  # type: ignore[arg-type]
                    _forward_block,
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                with pick_attention_backend(self.config.use_flash_attn):
                    hidden_states = block(hidden_states, key_padding_mask=key_padding_mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        self.eval()
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

# ----------------------------------------------------------------------
# Checkpoint helpers
# ----------------------------------------------------------------------
def _load_checkpoint_bytes(path: str) -> bytes:
    if path.startswith("gs://"):
        with open_sharded_file(path, "rb") as handle:
            return handle.read()
    with open(path, "rb") as handle:
        return handle.read()


def _infer_block_count(state_dict: Dict[str, torch.Tensor]) -> int:
    indices = set()
    prefix = "blocks."
    for key in state_dict:
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :]
        block_id, *_ = remainder.split(".")
        if block_id.isdigit():
            indices.add(int(block_id))
    return len(indices)


def _validate_checkpoint_shapes(
    model: Stage1Model,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    expected = model.state_dict()

    width_key = "token_embed.weight"
    if width_key not in state_dict:
        raise ValueError("Checkpoint is missing token embedding weights; incompatible with Stage-1 model")

    checkpoint_width = state_dict[width_key].shape[1]
    expected_width = expected[width_key].shape[1]
    if checkpoint_width != expected_width:
        raise ValueError(
            "Checkpoint hidden width mismatch: expected "
            f"{expected_width}, found {checkpoint_width}."
        )

    checkpoint_blocks = _infer_block_count(state_dict)
    expected_blocks = len(model.blocks)
    if checkpoint_blocks != expected_blocks:
        raise ValueError(
            "Checkpoint depth mismatch: expected "
            f"{expected_blocks} transformer blocks, found {checkpoint_blocks}."
        )

    mismatched: list[str] = []
    for key, tensor in state_dict.items():
        if key in expected and expected[key].shape != tensor.shape:
            mismatched.append(
                f"{key} (expected {tuple(expected[key].shape)}, got {tuple(tensor.shape)})"
            )
    if mismatched:
        raise ValueError(
            "Checkpoint parameter shape mismatch detected:\n" + "\n".join(sorted(mismatched))
        )


def load_stage1_checkpoint(
    model: Stage1Model,
    checkpoint_path: str,
    device: torch.device | str | None = None,
) -> CheckpointMetadata:
    """Load a Stage-1 checkpoint into ``model``.

    The loader validates the depth/width of the checkpoint before loading it in ``strict``
    mode. ``checkpoint_path`` may point to a local file or to a GCS URI.
    """

    raw = _load_checkpoint_bytes(checkpoint_path)
    buffer = io.BytesIO(raw)
    state = torch.load(buffer, map_location=device or "cpu")
    state_dict = state.get("state_dict", state)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint format is invalid: expected mapping state_dict")

    _validate_checkpoint_shapes(model, state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            "Checkpoint compatibility failure. Missing keys: "
            f"{sorted(missing)}; unexpected keys: {sorted(unexpected)}"
        )

    step = int(state.get("step", 0))
    metric_value = None
    metrics: Dict[str, float] = {}
    if isinstance(state.get("metrics"), dict):
        metrics = {k: float(v) for k, v in state["metrics"].items() if isinstance(v, (int, float))}
        metric_value = metrics.get("val_perplexity")

    optimizer_state = state.get("optimizer") if isinstance(state, dict) else None
    scheduler_state = state.get("scheduler") if isinstance(state, dict) else None

    return CheckpointMetadata(
        step=step,
        metric_value=metric_value,
        metrics=metrics,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
    )


__all__ = ["Stage1Model", "load_stage1_checkpoint", "CheckpointMetadata"]
