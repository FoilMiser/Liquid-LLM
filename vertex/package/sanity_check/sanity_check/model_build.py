"""Model reconstruction utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import nn

try:  # pragma: no cover - optional import
    from transformers import GPT2Config, GPT2LMHeadModel
except Exception:  # pragma: no cover - transformers is optional at runtime
    GPT2Config = None  # type: ignore
    GPT2LMHeadModel = None  # type: ignore


@dataclass
class StudentConfig:
    """Minimal GPT-like configuration."""

    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    bos_id: Optional[int] = None
    eos_id: Optional[int] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, block_size: int) -> "StudentConfig":
        return cls(
            vocab_size=int(data.get("vocab_size", 50257)),
            n_layer=int(data.get("n_layer", 12)),
            n_head=int(data.get("n_head", 12)),
            n_embd=int(data.get("n_embd", 768)),
            block_size=int(data.get("block_size", block_size)),
            bos_id=data.get("bos_id"),
            eos_id=data.get("eos_id"),
        )


def _infer_from_state_dict(state_dict: Mapping[str, torch.Tensor], *, block_size: int) -> StudentConfig:
    embed_weight = state_dict.get("transformer.wte.weight") or state_dict.get("tok_embeddings.weight")
    if embed_weight is None:
        raise ValueError("Unable to infer vocab size; missing embedding weight in state dict")
    vocab_size, n_embd = embed_weight.shape

    # Find attention weights to infer heads/layers
    n_layer = 0
    n_head = 0
    for name, tensor in state_dict.items():
        if "attn.c_attn.weight" in name or "attention.attention.weight" in name:
            n_layer = max(n_layer, int(name.split(".")[2]))
        if "attn.c_attn.weight" in name:
            n_head = tensor.shape[1] // n_embd
    if n_layer == 0:
        # fallback by counting blocks
        n_layer = sum(1 for name in state_dict if "mlp.c_fc.weight" in name or "mlp.fc_in.weight" in name)
    if n_layer == 0:
        n_layer = 12
    if n_head == 0:
        n_head = max(1, n_embd // 64)
    return StudentConfig(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(0.0)
        self.resid_dropout = nn.Dropout(0.0)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(mask == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTStudent(nn.Module):
    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(0.0)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError("Sequence length exceeds block size")
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return logits, loss


def build_model(state: Mapping[str, Any], *, block_size: int) -> Tuple[nn.Module, StudentConfig, Dict[str, Any]]:
    """Reconstruct the student model from checkpoint state."""

    model_cfg = state.get("model_cfg")
    if model_cfg is not None:
        config = StudentConfig.from_mapping(model_cfg, block_size=block_size)
    else:
        config = _infer_from_state_dict(state["state_dict"], block_size=block_size)

    extra: Dict[str, Any] = {}
    if GPT2LMHeadModel is not None:
        try:
            hf_config = GPT2Config(
                vocab_size=config.vocab_size,
                n_layer=config.n_layer,
                n_head=config.n_head,
                n_embd=config.n_embd,
                n_positions=config.block_size,
                n_ctx=config.block_size,
                bos_token_id=config.bos_id,
                eos_token_id=config.eos_id,
            )
            model = GPT2LMHeadModel(hf_config)
            extra["backend"] = "transformers"
            return model, config, extra
        except Exception:  # pragma: no cover - fallback to native implementation
            extra["backend"] = "native"
    else:
        extra["backend"] = "native"

    model = GPTStudent(config)
    return model, config, extra


def load_state_dict(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    """Attempt to load the state dict strictly, then fall back to non-strict."""

    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        return {"missing_keys": missing, "unexpected_keys": unexpected, "strict": True}
    except RuntimeError:
        info = model.load_state_dict(state_dict, strict=False)
        missing, unexpected = info
        return {"missing_keys": missing, "unexpected_keys": unexpected, "strict": False}
