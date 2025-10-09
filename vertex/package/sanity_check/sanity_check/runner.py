"""Orchestrates the sanity check flow."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from . import __version__
from .io_gcs import download_checkpoint
from .logutil import log_json, summarize
from .tests_env import check_environment
from .tests_model import (
    BatchData,
    ModelArtifacts,
    backward_smoke,
    forward_smoke,
    load_model_from_checkpoint,
    prepare_tokenizer_batch,
)
from .tests_perf import memory_stats, throughput_probe
from .types import RunContext, TestResult


DEFAULT_GCS_URI = "gs://liquid-llm-bucket-2/stage0/checkpoints/vertex_runs/20251009-022648/IMPORTANT/stage0_checkpoints_vertex_runs_20251009-022648_best.pt"


def _download_checkpoint(ctx: RunContext, uri: str) -> Tuple[TestResult, Optional[Path]]:
    start = time.perf_counter()
    status = "PASS"
    error: Optional[str] = None
    metrics = {"uri": uri}
    path: Optional[Path] = None

    try:
        target = Path("/tmp/model_sanity")
        path = download_checkpoint(uri, target)
        metrics["local_path"] = str(path)
        metrics["file_size_bytes"] = path.stat().st_size if path.exists() else None
    except Exception as exc:
        status = "FAIL"
        error = repr(exc)

    duration_ms = (time.perf_counter() - start) * 1000.0
    log_json("checkpoint_download", status, ctx, metrics=metrics, duration_ms=duration_ms, error=error)
    return TestResult(event="checkpoint_download", status=status, metrics=metrics, duration_ms=duration_ms, required=True, error=error), path


def run(args: argparse.Namespace) -> int:
    overall_start = time.perf_counter()
    device = torch.device(args.device)
    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    ctx = RunContext(
        device=str(device),
        cuda_capability="unknown",
        cuda_driver=torch.version.cuda or "cpu",
        torch_version=torch.__version__,
        run_id=run_id,
        pkg_version=__version__,
    )

    results: List[TestResult] = []

    def record(result: TestResult) -> None:
        results.append(result)

    def check_runtime() -> bool:
        elapsed = time.perf_counter() - overall_start
        if elapsed > args.max_time_s:
            metrics = {"elapsed_s": round(elapsed, 3), "max_time_s": args.max_time_s}
            log_json("runtime_limit", "FAIL", ctx, metrics=metrics, duration_ms=0.0)
            timeout_result = TestResult(event="runtime_limit", status="FAIL", metrics=metrics, duration_ms=0.0, required=True, error="Max runtime exceeded")
            record(timeout_result)
            summarize(ctx, tests_passed=sum(r.status == "PASS" for r in results),
                      tests_failed=sum(r.status != "PASS" for r in results),
                      total_duration_ms=(time.perf_counter() - overall_start) * 1000.0,
                      status="FAIL")
            return True
        return False

    # Environment check
    env_result = check_environment(ctx, args.device, args.seed)
    record(env_result)
    env_metrics = env_result.metrics
    if "cuda_capability" in env_metrics:
        ctx.cuda_capability = str(env_metrics.get("cuda_capability"))
    if "cuda_driver_version" in env_metrics:
        ctx.cuda_driver = str(env_metrics.get("cuda_driver_version"))

    if env_result.status != "PASS" and env_result.required:
        summarize(ctx, tests_passed=0, tests_failed=1, total_duration_ms=(time.perf_counter() - overall_start) * 1000.0, status="FAIL")
        return 2
    if check_runtime():
        return 2

    download_result, checkpoint_path = _download_checkpoint(ctx, args.checkpoint_gcs_uri)
    record(download_result)
    if download_result.status != "PASS" and download_result.required:
        summarize(ctx, tests_passed=sum(r.status == "PASS" for r in results),
                  tests_failed=sum(r.status != "PASS" for r in results),
                  total_duration_ms=(time.perf_counter() - overall_start) * 1000.0,
                  status="FAIL")
        return 2
    if check_runtime():
        return 2

    bundle: Optional[ModelArtifacts] = None
    tokenizer_batch: Optional[Tuple[PreTrainedTokenizerBase, BatchData]] = None

    if checkpoint_path is not None:
        model_result, bundle = load_model_from_checkpoint(
            ctx,
            checkpoint_path,
            device=device,
            block_size=args.block_size,
            dtype_name=args.dtype,
        )
        record(model_result)
    else:
        model_result = TestResult(event="model_load", status="FAIL", metrics={}, duration_ms=0.0, required=True, error="Checkpoint path missing")
        record(model_result)

    if bundle is None or model_result.status != "PASS":
        summarize(ctx, tests_passed=sum(r.status == "PASS" for r in results),
                  tests_failed=sum(r.status != "PASS" for r in results),
                  total_duration_ms=(time.perf_counter() - overall_start) * 1000.0,
                  status="FAIL")
        return 2
    if check_runtime():
        return 2

    tokenizer_result, tokenizer_batch = prepare_tokenizer_batch(
        ctx,
        tokenizer_name=args.tokenizer_name,
        block_size=args.block_size,
        batch_size=args.batch_size,
        device=device,
    )
    record(tokenizer_result)

    if tokenizer_batch is None:
        summarize(ctx, tests_passed=sum(r.status == "PASS" for r in results),
                  tests_failed=sum(r.status != "PASS" for r in results),
                  total_duration_ms=(time.perf_counter() - overall_start) * 1000.0,
                  status="FAIL")
        return 2
    if check_runtime():
        return 2

    _, batch = tokenizer_batch
    forward_result = forward_smoke(ctx, bundle.model, batch, dtype_name=args.dtype)
    record(forward_result)
    if forward_result.status != "PASS" and forward_result.required:
        summarize(ctx, tests_passed=sum(r.status == "PASS" for r in results),
                  tests_failed=sum(r.status != "PASS" for r in results),
                  total_duration_ms=(time.perf_counter() - overall_start) * 1000.0,
                  status="FAIL")
        return 2
    if check_runtime():
        return 2

    backward_result = backward_smoke(ctx, bundle.model, batch, dtype_name=args.dtype)
    record(backward_result)
    if backward_result.status != "PASS" and backward_result.required:
        summarize(ctx, tests_passed=sum(r.status == "PASS" for r in results),
                  tests_failed=sum(r.status != "PASS" for r in results),
                  total_duration_ms=(time.perf_counter() - overall_start) * 1000.0,
                  status="FAIL")
        return 2
    if check_runtime():
        return 2

    throughput_result = throughput_probe(
        ctx,
        bundle.model,
        total_tokens=args.throughput_tokens,
        batch_size=args.batch_size,
        block_size=args.block_size,
        dtype_name=args.dtype,
    )
    record(throughput_result)
    if check_runtime():
        return 2

    memory_result = memory_stats(ctx)
    record(memory_result)

    total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
    tests_passed = sum(r.status == "PASS" for r in results)
    tests_failed = sum(r.status != "PASS" for r in results)
    required_failed = any(r.required and r.status != "PASS" for r in results)

    status = "FAIL" if required_failed else "PASS"
    summarize(ctx, tests_passed=tests_passed, tests_failed=tests_failed, total_duration_ms=total_duration_ms, status=status)
    return 2 if required_failed else 0
