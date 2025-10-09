"""Environment and GPU checks."""

from __future__ import annotations

import platform
import time
from typing import Dict, Optional

import torch

from .logutil import log_json
from .types import RunContext, TestResult


REQUIRED_TESTS = {"env_check"}


def check_environment(ctx: RunContext, device: str, seed: int) -> TestResult:
    start = time.perf_counter()
    metrics: Dict[str, object] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "torch_cuda_available": torch.cuda.is_available(),
    }
    error: Optional[str] = None
    status = "PASS"

    try:
        device_obj = torch.device(device)
        if device_obj.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
            torch.cuda.set_device(device_obj)
            capability = torch.cuda.get_device_capability(device_obj)
            metrics["cuda_device_name"] = torch.cuda.get_device_name(device_obj)
            metrics["cuda_capability"] = f"{capability[0]}.{capability[1]}"
            metrics["cuda_driver_version"] = torch.version.cuda
            metrics["cudnn_version"] = torch.backends.cudnn.version()
            metrics["allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
            metrics["supports_bf16"] = torch.cuda.is_bf16_supported()
            metrics["supports_fp16"] = torch.cuda.is_available()
            torch.cuda.empty_cache()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
    except Exception as exc:
        status = "FAIL"
        error = repr(exc)

    duration_ms = (time.perf_counter() - start) * 1000.0
    log_json("env_check", status, ctx, metrics=metrics, duration_ms=duration_ms, error=error)
    return TestResult(event="env_check", status=status, metrics=metrics, duration_ms=duration_ms, required=True, error=error)
