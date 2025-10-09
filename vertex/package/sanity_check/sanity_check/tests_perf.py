"""Performance and memory probes."""

from __future__ import annotations

import math
import time
from typing import Dict, Optional

import torch

from .logutil import log_json
from .types import RunContext, TestResult
from .tests_model import _autocast, _select_dtype, _unwrap_outputs


def throughput_probe(ctx: RunContext, model: torch.nn.Module, *, total_tokens: int, batch_size: int,
                     block_size: int, dtype_name: str) -> TestResult:
    start = time.perf_counter()
    metrics: Dict[str, object] = {
        "total_tokens_target": total_tokens,
        "batch_size": batch_size,
        "block_size": block_size,
    }
    status = "PASS"
    error: Optional[str] = None

    try:
        device = next(model.parameters()).device
        dtype = _select_dtype(dtype_name)
        seq_len = min(block_size, 256)
        tokens_per_batch = batch_size * seq_len
        steps = max(1, math.ceil(total_tokens / tokens_per_batch))
        input_ids = torch.randint(low=0, high=model.config.vocab_size if hasattr(model, "config") else 50257,
                                  size=(batch_size, seq_len), device=device)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        model.eval()
        # Warmup
        with torch.no_grad():
            with _autocast(device, dtype):
                model(input_ids, labels=labels)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies = []
        processed_tokens = 0
        for _ in range(steps):
            if time.perf_counter() - start > 170.0:
                status = "FAIL"
                error = "Throughput probe exceeded time budget"
                break
            begin = time.perf_counter()
            with torch.no_grad():
                with _autocast(device, dtype):
                    outputs = model(input_ids, labels=labels)
                    logits, loss = _unwrap_outputs(outputs)
                    del logits, loss
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency = time.perf_counter() - begin
            latencies.append(latency)
            processed_tokens += tokens_per_batch
        if latencies:
            latencies_sorted = sorted(latencies)
            mid = len(latencies_sorted) // 2
            p50 = latencies_sorted[mid]
            p90 = latencies_sorted[int(0.9 * (len(latencies_sorted) - 1))]
            total_time = sum(latencies)
            tokens_sec = processed_tokens / total_time if total_time > 0 else 0.0
            metrics.update({
                "steps": len(latencies),
                "processed_tokens": processed_tokens,
                "tokens_per_second": tokens_sec,
                "latency_p50_ms": p50 * 1000.0,
                "latency_p90_ms": p90 * 1000.0,
            })
        else:
            metrics["steps"] = 0
            status = "FAIL"
            error = error or "No throughput steps executed"
    except Exception as exc:
        status = "FAIL"
        error = repr(exc)

    duration_ms = (time.perf_counter() - start) * 1000.0
    log_json("throughput", status, ctx, metrics=metrics, duration_ms=duration_ms, error=error)
    return TestResult(event="throughput", status=status, metrics=metrics, duration_ms=duration_ms, required=False, error=error)


def memory_stats(ctx: RunContext) -> TestResult:
    start = time.perf_counter()
    metrics: Dict[str, object] = {}
    status = "PASS"
    error: Optional[str] = None

    try:
        if not torch.cuda.is_available():
            metrics["message"] = "CUDA not available; memory stats skipped"
        else:
            device_idx = torch.cuda.current_device()
            allocated = torch.cuda.max_memory_allocated(device_idx) / (1024 ** 2)
            reserved = torch.cuda.max_memory_reserved(device_idx) / (1024 ** 2)
            metrics.update({
                "max_memory_allocated_mib": round(allocated, 2),
                "max_memory_reserved_mib": round(reserved, 2),
            })
            torch.cuda.reset_peak_memory_stats(device_idx)
    except Exception as exc:
        status = "FAIL"
        error = repr(exc)

    duration_ms = (time.perf_counter() - start) * 1000.0
    log_json("memory_stats", status, ctx, metrics=metrics, duration_ms=duration_ms, error=error)
    return TestResult(event="memory_stats", status=status, metrics=metrics, duration_ms=duration_ms, required=False, error=error)
