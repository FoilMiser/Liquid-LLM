"""Compatibility package exposing the Stage-1 Vertex entrypoint."""

from __future__ import annotations

from Stage_1.vertex import entrypoint as _entrypoint

__all__ = getattr(_entrypoint, "__all__", tuple())

for _name in __all__:
    globals()[_name] = getattr(_entrypoint, _name)

if "main" not in __all__ and hasattr(_entrypoint, "main"):
    main = _entrypoint.main  # type: ignore[assignment]
    __all__ = tuple(__all__) + ("main",)

# Re-export the entrypoint module so legacy import paths continue to function.
entrypoint = _entrypoint
