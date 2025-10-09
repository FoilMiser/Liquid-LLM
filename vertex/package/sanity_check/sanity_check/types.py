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


class TestResult(TypedDict):
    """Result of an individual test."""

    event: str
    status: str
    metrics: Dict[str, Any]
    duration_ms: float
    required: bool
    error: Optional[str]


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
