"""Student model initialisation utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcs_io import gcs_to_local
from .runtime_setup import enable_flash_attn_if_available
from .utils import configure_logging, ensure_dir

logger = configure_logging()


@dataclass
class StudentConfig:
    vocab_size: int
    hidden_size: int
    n_heads: int
    n_layers: int
    intermediate_size: int
    sequence_length: int = 1024


class MLP(nn.Module):
    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_size // config.n_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = attn.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class StudentModel(nn.Module):
    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


_FROZEN_BLOCK_NAMES = ("block_0", "block_1")


def infer_config_from_state_dict(state_dict: Dict[str, torch.Tensor], seq_len: int) -> StudentConfig:
    """Infer a transformer configuration from a state dict."""

    vocab_size = 32000
    for key, tensor in state_dict.items():
        if "embed" in key and tensor.ndim == 2:
            vocab_size = tensor.shape[0]
            break
    hidden_size = next(iter(state_dict.values())).shape[-1]
    n_layers = len({key.split(".")[2] for key in state_dict if key.startswith("blocks.")})
    n_heads = 16
    for key, tensor in state_dict.items():
        if key.endswith("q_proj.weight"):
            hidden_size = tensor.shape[0]
            n_heads = max(1, tensor.shape[0] // tensor.shape[1])
            break
    intermediate_size = hidden_size * 4
    return StudentConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_layers=n_layers,
        intermediate_size=intermediate_size,
        sequence_length=seq_len,
    )


def build_student_model_from_state(state_dict: Dict[str, torch.Tensor], seq_len: int = 1024) -> StudentModel:
    config = infer_config_from_state_dict(state_dict, seq_len)
    model = StudentModel(config)
    model.load_state_dict(state_dict, strict=False)
    return model


def load_student_from_gcs(gcs_uri: str, seq_len: int) -> Tuple[StudentModel, Dict[str, torch.Tensor]]:
    """Download a student checkpoint from GCS and return the model and state."""

    local_path = gcs_to_local(gcs_uri, "/tmp/student_checkpoint.pt")
    state_dict = torch.load(local_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model = build_student_model_from_state(state_dict, seq_len=seq_len)
    return model, state_dict


def save_frozen_mask(run_dir: str, block_names: Iterable[str], *, output_gcs_uri: str | None = None, run_id: str | None = None) -> Path:
    ensure_dir(run_dir)
    path = Path(run_dir) / "frozen_mask.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump({"trainable_blocks": list(block_names)}, f, indent=2, sort_keys=True)
    if output_gcs_uri and run_id:
        from .gcs_io import local_to_gcs

        dest = f"{output_gcs_uri.rstrip('/')}/{run_id}/frozen_mask.json"
        local_to_gcs(str(path), dest)
    return path


def initialize_student(
    gcs_uri: str,
    run_dir: str,
    seq_len: int,
    *,
    output_gcs_uri: str | None = None,
    run_id: str | None = None,
) -> StudentModel:
    model, _ = load_student_from_gcs(gcs_uri, seq_len)
    enable_flash_attn_if_available()
    save_frozen_mask(run_dir, _FROZEN_BLOCK_NAMES, output_gcs_uri=output_gcs_uri, run_id=run_id)
    return model


def apply_grad_checkpointing(model: nn.Module, enabled: bool) -> nn.Module:
    if not enabled:
        return model
    import torch.utils.checkpoint as cp

    for module in model.modules():
        name = module.__class__.__name__.lower()
        if name.endswith("block") or name.endswith("layer"):
            if getattr(module, "_checkpoint_wrapped", False):
                continue
            orig_forward = module.forward

            def checkpointed_forward(*args, _orig=orig_forward, **kwargs):
                return cp.checkpoint(_orig, *args, **kwargs)

            module.forward = checkpointed_forward  # type: ignore[assignment]
            setattr(module, "_checkpoint_wrapped", True)
    return model
