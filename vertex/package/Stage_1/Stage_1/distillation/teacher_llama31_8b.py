"""Teacher wrapper for llama-3.1-8b."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None

from ..utils.secrets import get_secret


@dataclass
class TeacherConfig:
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    endpoint: Optional[str] = None
    hf_secret_name: Optional[str] = None
    device: str = "cuda"


class TeacherModel:
    def __init__(self, config: TeacherConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        if config.endpoint:
            raise NotImplementedError("Remote inference endpoint integration pending")
        if AutoModelForCausalLM is None:
            raise ImportError("transformers package required for teacher model")
        token = get_secret(config.hf_secret_name) if config.hf_secret_name else None
        auth = {"use_auth_token": token} if token else {}
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, **auth)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16, device_map="auto", **auth)
        self.model.eval()

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert self.model is not None
        outputs = self.model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device) if attention_mask is not None else None)
        return outputs.logits.to(input_ids.device)


__all__ = ["TeacherModel", "TeacherConfig"]
