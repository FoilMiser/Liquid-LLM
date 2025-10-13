"""Stage-1 Vertex AI training package for Liquid LLM."""

from __future__ import annotations

from . import cli as _cli  # noqa: F401 (re-export for entry point convenience)

__all__ = ["_cli"]
