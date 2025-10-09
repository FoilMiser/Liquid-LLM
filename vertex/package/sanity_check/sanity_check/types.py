"""Shared type definitions for sanity check results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, TypedDict


class LogRecord(TypedDict, total=False):
    """Structure of a JSON log record."""

    event: str
    status: str
    metrics: Mapping[str, Any]
    duration_ms: float
    message: str
    error: str
    device: str
    cuda_capability: str
    cuda_driver: str
    torch_version: str
    run_id: str
    pkg_version: str


@dataclass
class TestResult:
    """Result of an individual test."""

    event: str
    status: str
    metrics: Dict[str, Any]
    duration_ms: float
    required: bool
    error: Optional[str]

    def is_pass(self) -> bool:
        """Return True when the test finished successfully."""

        return self.status == "PASS"

    def to_summary(self) -> Dict[str, Any]:
        """Return a compact summary suitable for logging."""

        summary: Dict[str, Any] = {
            "event": self.event,
            "status": self.status,
            "required": self.required,
            "duration_ms": round(float(self.duration_ms), 3),
        }
        if self.error:
            summary["error"] = self.error
        return summary


@dataclass
class RunContext:
    """Runtime context shared across tests."""

    device: str
    cuda_capability: str
    cuda_driver: str
    torch_version: str
    run_id: str
    pkg_version: str


class TensorSpec(TypedDict):
    """Minimal tensor specification used when inferring model config."""

    name: str
    shape: Iterable[int]
    dtype: str
