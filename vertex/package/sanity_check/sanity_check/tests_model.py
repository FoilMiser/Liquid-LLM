"""Model-centric sanity tests."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .io_gcs import hash_state_dict
from .logutil import log_json
from .model_build import StudentConfig, build_model, load_state_dict
from .types import RunContext, TestResult


REQUIRED_TESTS = {"model_load", "forward_smoke", "backward_smoke"}


@dataclass
class ModelArtifacts:
    model: nn.Module
    config: StudentConfig
    checkpoint: Dict[str, object]
    load_info: Dict[str, object]
    param_count: int
    state_hash: str


@dataclass
class BatchData:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    seq_len: int
    batch_size: int


SAMPLE_TEXTS = [
    "The distilled Stage-0 model should emit coherent tokens.",
    "Vertex AI jobs require deterministic sanity checks.",
    "Quick brown fox jumps over the lazy dog.",
    "1234567890 symbols and punctuation!",
    "Mixing short and longer sentences ensures padding works.",
]


def _select_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _autocast(device: torch.device, dtype: torch.dtype):
    if dtype == torch.float32:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _unwrap_outputs(outputs: object) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(outputs, tuple):
        logits = outputs[0]
        loss = outputs[1] if len(outputs) > 1 else None
        return logits, loss
    if hasattr(outputs, "logits"):
        logits = outputs.logits
        loss = getattr(outputs, "loss", None)
        return logits, loss
    raise TypeError("Unexpected model output type")


def load_model_from_checkpoint(ctx: RunContext, checkpoint_path: Path, *, device: torch.device,
                               block_size: int, dtype_name: str) -> Tuple[TestResult, Optional[ModelArtifacts]]:
    start = time.perf_counter()
    metrics: Dict[str, object] = {}
    error: Optional[str] = None
    status = "PASS"
    bundle: Optional[ModelArtifacts] = None

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" not in checkpoint:
            raise KeyError("Checkpoint missing 'state_dict'")
        state_dict = checkpoint["state_dict"]
        state_hash = hash_state_dict(state_dict)
        param_count = int(sum(t.numel() for t in state_dict.values()))
        model, config, extra = build_model(checkpoint, block_size=block_size)
        load_info = load_state_dict(model, state_dict)
        model.to(device)
        dtype = _select_dtype(dtype_name)
        if dtype != torch.float32:
            if device.type != "cuda":
                metrics["dtype_warning"] = "Non-float32 dtype requested on CPU; keeping float32"
            else:
                model.to(dtype=dtype)
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        metrics.update({
            "param_count": param_count,
            "state_hash": state_hash,
            "backend": extra.get("backend", "native"),
            "strict_load": load_info.get("strict", False),
            "missing_keys": len(load_info.get("missing_keys", [])),
            "unexpected_keys": len(load_info.get("unexpected_keys", [])),
        })
        bundle = ModelArtifacts(
            model=model,
            config=config,
            checkpoint=checkpoint,
            load_info=load_info,
            param_count=param_count,
            state_hash=state_hash,
        )
    except Exception as exc:
        status = "FAIL"
        error = repr(exc)

    duration_ms = (time.perf_counter() - start) * 1000.0
    log_json("model_load", status, ctx, metrics=metrics, duration_ms=duration_ms, error=error)
    return TestResult(event="model_load", status=status, metrics=metrics, duration_ms=duration_ms, required=True, error=error), bundle


def prepare_tokenizer_batch(ctx: RunContext, tokenizer_name: str, block_size: int, batch_size: int,
                             device: torch.device) -> Tuple[TestResult, Optional[Tuple[PreTrainedTokenizerBase, BatchData]]]:
    start = time.perf_counter()
    metrics: Dict[str, object] = {}
    status = "PASS"
    error: Optional[str] = None
    result: Optional[Tuple[PreTrainedTokenizerBase, BatchData]] = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer(SAMPLE_TEXTS[:batch_size], padding=True, truncation=True,
                            max_length=block_size, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        seq_len = int(attention_mask.sum(dim=1).max().item())
        pad_ratio = 1.0 - float(attention_mask.sum().item() / (attention_mask.numel()))
        metrics.update({
            "batch_size": int(input_ids.size(0)),
            "seq_len_max": seq_len,
            "pad_ratio": round(pad_ratio, 4),
            "tokenizer_name": tokenizer_name,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        })
        result = (tokenizer, BatchData(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            seq_len=seq_len,
            batch_size=input_ids.size(0),
        ))
    except Exception as exc:
        status = "FAIL"
        error = repr(exc)

    duration_ms = (time.perf_counter() - start) * 1000.0
    log_json("tokenizer_roundtrip", status, ctx, metrics=metrics, duration_ms=duration_ms, error=error)
    return TestResult(event="tokenizer_roundtrip", status=status, metrics=metrics, duration_ms=duration_ms, required=False, error=error), result


def forward_smoke(ctx: RunContext, model: nn.Module, batch: BatchData, *, dtype_name: str) -> TestResult:
    start = time.perf_counter()
    metrics: Dict[str, object] = {}
    status = "PASS"
    error: Optional[str] = None
    dtype = _select_dtype(dtype_name)

    try:
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            with _autocast(device, dtype):
                outputs = model(batch.input_ids.to(device), labels=batch.labels.to(device))
        logits, loss = _unwrap_outputs(outputs)
        metrics["logits_shape"] = list(logits.shape)
        metrics["loss"] = float(loss.item()) if loss is not None else None
        metrics["loss_isfinite"] = bool(torch.isfinite(loss)) if loss is not None else True
        metrics["has_nan"] = bool(torch.isnan(logits).any().item())
        if loss is not None and not torch.isfinite(loss):
            raise RuntimeError("Loss is not finite")
        if torch.isnan(logits).any():
            raise RuntimeError("Logits contain NaN")
    except Exception as exc:
        status = "FAIL"
        error = repr(exc)

    duration_ms = (time.perf_counter() - start) * 1000.0
    log_json("forward_smoke", status, ctx, metrics=metrics, duration_ms=duration_ms, error=error)
    return TestResult(event="forward_smoke", status=status, metrics=metrics, duration_ms=duration_ms, required=True, error=error)


def backward_smoke(ctx: RunContext, model: nn.Module, batch: BatchData, *, dtype_name: str) -> TestResult:
    start = time.perf_counter()
    metrics: Dict[str, object] = {}
    status = "PASS"
    error: Optional[str] = None
    dtype = _select_dtype(dtype_name)

    try:
        device = next(model.parameters()).device
        model.train()
        size = min(4, batch.batch_size)
        seq = min(batch.seq_len, 64)
        small_batch = BatchData(
            input_ids=batch.input_ids[:size, :seq].to(device),
            attention_mask=batch.attention_mask[:size, :seq].to(device),
            labels=batch.labels[:size, :seq].to(device),
            seq_len=seq,
            batch_size=size,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, dtype):
            outputs = model(small_batch.input_ids, labels=small_batch.labels)
        logits, loss = _unwrap_outputs(outputs)
        if loss is None:
            raise RuntimeError("Model did not return a loss")
        if not torch.isfinite(loss):
            raise RuntimeError("Loss is not finite")
        loss.backward()
        grad_finite = True
        for param in model.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                grad_finite = False
                break
        if not grad_finite:
            raise RuntimeError("Detected non-finite gradients")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metrics["loss"] = float(loss.item())
        metrics["batch_tokens"] = int(small_batch.batch_size * small_batch.seq_len)
    except Exception as exc:
        status = "FAIL"
        error = repr(exc)

    duration_ms = (time.perf_counter() - start) * 1000.0
    log_json("backward_smoke", status, ctx, metrics=metrics, duration_ms=duration_ms, error=error)
    return TestResult(event="backward_smoke", status=status, metrics=metrics, duration_ms=duration_ms, required=True, error=error)
