"""Stage-1 Vertex AI training package for Liquid LLM."""

from __future__ import annotations

import sys
import types
from importlib import import_module

from . import cli as _cli  # noqa: F401 (re-export for entry point convenience)

__all__ = ["_cli"]


def _install_trainer_compatibility() -> None:
    """Expose the historic ``trainer`` module path for Vertex AI jobs."""
    if "trainer.entrypoint" in sys.modules:
        return

    entrypoint_module = import_module(".vertex.entrypoint", __name__)

    shim = types.ModuleType("trainer")
    shim.__path__ = []  # type: ignore[attr-defined]
    shim.entrypoint = entrypoint_module

    main = getattr(entrypoint_module, "main", None)
    if main is not None:
        shim.main = main
        shim.__all__ = ["entrypoint", "main"]
    else:  # pragma: no cover - defensive fallback
        shim.__all__ = ["entrypoint"]

    sys.modules["trainer"] = shim
    sys.modules["trainer.entrypoint"] = entrypoint_module


_install_trainer_compatibility()
