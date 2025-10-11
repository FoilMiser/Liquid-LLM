"""CLI for Stage-1 training."""

from __future__ import annotations

from typing import Sequence

from .vertex.entrypoint import main as _entrypoint_main


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate to the Vertex entrypoint for backwards compatibility."""

    result = _entrypoint_main(argv)
    return 0 if result is None else result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
