"""Logging helpers for Stage-1."""

from __future__ import annotations

import logging
from typing import Optional

try:  # pragma: no cover - optional dependency guard
    from pythonjsonlogger import jsonlogger
except Exception:  # pragma: no cover - formatter fallback
    jsonlogger = None  # type: ignore[assignment]


def _build_formatter() -> logging.Formatter:
    if jsonlogger is not None:
        return jsonlogger.JsonFormatter("%(levelname)s %(name)s %(message)s")
    return logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the Stage-1 package logger."""

    root = logging.getLogger("Stage_1")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_build_formatter())
        root.addHandler(handler)
        root.propagate = False
    root.setLevel(level)
    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger scoped under the Stage-1 namespace."""

    root = configure_logging()
    if not name:
        return root
    return root.getChild(name)


__all__ = ["configure_logging", "get_logger"]
