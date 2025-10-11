"""Lightweight fallback for :mod:`pythonjsonlogger` when the dependency is missing.

Vertex AI bootstrapping sometimes imports :mod:`pythonjsonlogger` via ``sitecustomize``
*before* installing package dependencies.  That fails noisily unless the module is
already importable.  We ship a tiny, compatible subset of the library so the import
succeeds even in the bootstrap phase.  Once the real dependency is installed our
compatibility layer is still good enough for the Stage-1 logging needs.
"""

from __future__ import annotations

from . import jsonlogger

JsonFormatter = jsonlogger.JsonFormatter

__all__ = ["JsonFormatter", "jsonlogger"]
