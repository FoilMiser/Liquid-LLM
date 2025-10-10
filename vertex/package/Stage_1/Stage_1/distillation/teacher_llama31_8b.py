"""Teacher wrapper for Meta Llama 3.1 8B used in Stage-1 distillation."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

import torch

try:  # pragma: no cover - optional dependency in runtime container
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None

from ..utils.secrets import get_hf_token


LOGGER = logging.getLogger("Stage1Teacher")


@dataclass
class TeacherConfig:
    """Configuration for the Stage-1 teacher model."""

    model_id: str = "meta-llama/Meta-Llama-3.1-8B"
    endpoint: Optional[str] = None
    hf_secret_name: Optional[str] = None
    hf_token: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    device: str = "cuda"
    http_timeout: float = 30.0
    max_batch_size: int = 0


class TeacherModel:
    """Provides batched logits either via HF models or an HTTP endpoint."""

    def __init__(self, config: TeacherConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self._endpoint = config.endpoint
        self._max_batch_size = max(0, int(config.max_batch_size))

        if self._endpoint:
            LOGGER.info("Using teacher HTTP endpoint", extra={"endpoint": self._endpoint})
            self.model = None
            self.tokenizer = None
            return

        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError("transformers package is required for local teacher inference")

        token = config.hf_token or get_hf_token(config.hf_secret_name)
        auth_kwargs = {}
        if token:
            auth_kwargs["use_auth_token"] = token

        load_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": "auto" if self.device.type == "cuda" else None,
        }
        if config.hf_cache_dir:
            load_kwargs["cache_dir"] = config.hf_cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, **auth_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            **auth_kwargs,
            **load_kwargs,
        )
        self.model.requires_grad_(False)
        self.model.eval()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._endpoint:
            return self._logits_via_http(input_ids, attention_mask)

        assert self.model is not None, "Local teacher model was not initialised"
        model_device = next(self.model.parameters()).device
        original_device = input_ids.device

        batches = list(_chunk_tensor(input_ids, self._max_batch_size))
        masks = list(_chunk_tensor(attention_mask, self._max_batch_size)) if attention_mask is not None else [None] * len(batches)

        logits: list[torch.Tensor] = []
        for batch_ids, batch_mask in zip(batches, masks):
            batch_ids = batch_ids.to(model_device, non_blocking=True)
            batch_mask = batch_mask.to(model_device, non_blocking=True) if batch_mask is not None else None
            outputs = self.model(input_ids=batch_ids, attention_mask=batch_mask)
            logits.append(outputs.logits.to(original_device, dtype=torch.float32))

        return torch.cat(logits, dim=0)

    # ------------------------------------------------------------------
    def _logits_via_http(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        payload = {
            "input_ids": input_ids.cpu().tolist(),
            "attention_mask": attention_mask.cpu().tolist() if attention_mask is not None else None,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.http_timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:  # pragma: no cover - network failure path
            raise RuntimeError(f"Teacher endpoint request failed: {exc}") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Teacher endpoint returned non-JSON response") from exc

        if "logits" not in parsed:
            raise RuntimeError("Teacher endpoint response missing 'logits' field")

        tensor = torch.tensor(parsed["logits"], dtype=torch.float32)
        return tensor.to(input_ids.device)


def _chunk_tensor(tensor: Optional[torch.Tensor], max_batch_size: int) -> list[torch.Tensor]:
    if tensor is None:
        return []
    if max_batch_size <= 0 or tensor.size(0) <= max_batch_size:
        return [tensor]
    return [tensor[idx : idx + max_batch_size] for idx in range(0, tensor.size(0), max_batch_size)]


__all__ = ["TeacherModel", "TeacherConfig"]
