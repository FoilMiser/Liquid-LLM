"""Structured JSON logging utilities."""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

from .types import LogRecord, RunContext


@dataclass
class Timer:
    """Simple context manager timer."""

    start: float

    @classmethod
    @contextmanager
    def measure(cls) -> Iterator["Timer"]:
        start = time.perf_counter()
        timer = cls(start=start)
        try:
            yield timer
        finally:
            timer.start = (time.perf_counter() - start) * 1000.0

    @property
    def duration_ms(self) -> float:
        return self.start


def _base_record(event: str, ctx: RunContext) -> Dict[str, Any]:
    return {
        "event": event,
        "device": ctx.device,
        "cuda_capability": ctx.cuda_capability,
        "cuda_driver": ctx.cuda_driver,
        "torch_version": ctx.torch_version,
        "run_id": ctx.run_id,
        "pkg_version": ctx.pkg_version,
    }


def log_json(event: str, status: str, ctx: RunContext, *, metrics: Optional[Dict[str, Any]] = None,
             duration_ms: Optional[float] = None, message: Optional[str] = None,
             error: Optional[str] = None) -> None:
    """Emit a JSON record to stdout."""

    record: LogRecord = _base_record(event, ctx)  # type: ignore[assignment]
    record["status"] = status
    if metrics is not None:
        record["metrics"] = metrics
    if duration_ms is not None:
        record["duration_ms"] = float(duration_ms)
    if message:
        record["message"] = message
    if error:
        record["error"] = error
    json.dump(record, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


def timed_log(event: str, ctx: RunContext) -> Iterator[Timer]:
    """Context manager that yields a timer and automatically logs duration."""

    timer_start = time.perf_counter()
    timer = Timer(start=0.0)

    try:
        yield timer
    finally:
        duration = (time.perf_counter() - timer_start) * 1000.0
        timer.start = duration


def summarize(ctx: RunContext, tests_passed: int, tests_failed: int, total_duration_ms: float,
              extra_metrics: Optional[Dict[str, Any]] = None, status: str = "PASS") -> None:
    """Emit final summary log and human readable status."""

    metrics: Dict[str, Any] = {
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "total_duration_ms": total_duration_ms,
    }
    if extra_metrics:
        metrics.update(extra_metrics)

    log_json("summary", status, ctx, metrics=metrics, duration_ms=total_duration_ms)
    sys.stdout.write(f"FINAL_STATUS={status}\n")
    sys.stdout.flush()
