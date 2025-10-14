"""Minimal JSON logging formatter compatible with :mod:`python-json-logger`.

The real project offers many knobs.  For Stage-1 we only need deterministic JSON
strings so the Vertex AI bootstrapper can import :mod:`pythonjsonlogger` safely.
The implementation intentionally mirrors the public ``JsonFormatter`` API that the
Stage-1 training stack expects.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

__all__ = ["JsonFormatter"]


def _default(obj: Any) -> Any:
    """Fallback ``json.dumps`` serializer that stringifies unknown objects."""
    try:
        return str(obj)
    except Exception:  # pragma: no cover - very defensive
        return "<unserializable>"


class JsonFormatter(logging.Formatter):
    """Tiny subset of :class:`pythonjsonlogger.jsonlogger.JsonFormatter`."""

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: str = "%",
        ensure_ascii: bool = False,
        indent: int | None = None,
        rename_fields: Mapping[str, str] | None = None,
        mixin: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
        json_default: Callable[[Any], Any] | None = None,
        **json_kwargs: Any,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self._rename_fields = dict(rename_fields or {})
        self._mixin = mixin
        self._json_kwargs = {
            "ensure_ascii": ensure_ascii,
            "indent": indent,
            "default": json_default or _default,
        }
        self._json_kwargs.update(json_kwargs)

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: Mapping[str, Any],
    ) -> None:
        """Populate default fields on ``log_record`` in-place."""

        if record.exc_info:
            log_record.setdefault("exc_info", self.formatException(record.exc_info))
        if record.stack_info:
            log_record.setdefault("stack_info", self.formatStack(record.stack_info))

        log_record.setdefault("name", record.name)
        log_record.setdefault("levelname", record.levelname)
        log_record.setdefault("message", record.getMessage())
        # Vertex AI's sitecustomize expects "created" to exist
        log_record.setdefault("created", getattr(record, "created", time.time()))
        log_record.update(message_dict)

    def process_log_record(self, log_record: dict[str, Any]) -> dict[str, Any]:
        """Apply field renames and optional mixin payloads."""

        for source, destination in list(self._rename_fields.items()):
            if source in log_record:
                value = log_record.pop(source)
                if destination:
                    log_record[destination] = value

        if self._mixin:
            extra: Mapping[str, Any]
            if callable(self._mixin):
                extra = self._mixin() or {}
            else:
                extra = self._mixin
            log_record.update(dict(extra))

        return log_record

    def format(self, record: logging.LogRecord) -> str:
        message_dict: Mapping[str, Any]
        if isinstance(record.msg, Mapping):
            message_dict = dict(record.msg)
            # Ensure ``record.getMessage()`` does not try to format dicts with args.
            record.message = None  # type: ignore[attr-defined]
        else:
            message_dict = {}

        log_record: dict[str, Any] = {}
        self.add_fields(log_record, record, message_dict)
        processed = self.process_log_record(log_record)
        return json.dumps(processed, **self._json_kwargs)
