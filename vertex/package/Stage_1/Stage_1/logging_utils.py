"""Logging helpers for Stage-1."""

from __future__ import annotations

import logging
from importlib import import_module
from types import ModuleType
from typing import Optional


def _load_jsonlogger() -> Optional[ModuleType]:  # pragma: no cover - import shim
    """Return the vendored :mod:`pythonjsonlogger` module if available."""

    candidates = (
        "Stage_1.pythonjsonlogger.jsonlogger",
        "pythonjsonlogger.jsonlogger",
        "pythonjsonlogger",
    )
    for path in candidates:
        try:
            module = import_module(path)
        except Exception:
            continue
        if hasattr(module, "JsonFormatter"):
            return module
    return None


_JSONLOGGER = _load_jsonlogger()


def _build_formatter() -> logging.Formatter:
    if _JSONLOGGER is not None:
        try:
            return _JSONLOGGER.JsonFormatter("%(levelname)s %(name)s %(message)s")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - fall back to plain text formatter
            pass
    return logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")


def _ensure_stream_handler(logger: logging.Logger) -> logging.StreamHandler:
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            return handler

    handler = logging.StreamHandler()
    logger.addHandler(handler)
    return handler


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the Stage-1 package logger."""

    root = logging.getLogger("Stage_1")
    stream_handler = _ensure_stream_handler(root)
    stream_handler.setFormatter(_build_formatter())
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
