"""Utilities for structured logging used across the package."""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, Optional

_LOGGER: Optional[logging.Logger] = None


def configure_logging(run_id: str, level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger with a concise format.

    The configuration is idempotent and safe to call multiple times.
    """
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger("liquid_llm")
    logger.setLevel(level)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))

    logger.handlers.clear()
    logger.addHandler(handler)

    logger.info("Logging initialised for run_id=%s", run_id)
    _LOGGER = logger
    return logger


def log_info(message: str, **payload: Any) -> None:
    """Emit a JSON-formatted info message."""
    logger = _LOGGER or logging.getLogger("liquid_llm")
    if payload:
        message = f"{message} | {json.dumps(payload, sort_keys=True)}"
    logger.info(message)


def ensure_dir(path: str) -> str:
    """Create directory if it does not exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def safe_write_jsonl(path: str, records: Any) -> None:
    """Write a JSONL file atomically."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(tmp_path, path)


def git_sha_or_unknown() -> str:
    """Return the Git SHA if available, else 'unknown'."""
    try:
        import subprocess

        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd())
            .decode("utf-8")
            .strip()
        )
        return sha
    except Exception:  # pragma: no cover - defensive
        return "unknown"
