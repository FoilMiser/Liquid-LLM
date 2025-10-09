"""Vertex trainer compatibility entrypoint for the sanity-check package.

Vertex AI jobs that were originally configured to launch
``python -m trainer.entrypoint`` will import this module.  To keep
backwards compatibility with those job specifications we simply delegate
all argument parsing and execution to ``sanity_check.cli``.
"""

from __future__ import annotations

import sys
from typing import Sequence

from sanity_check import cli


def main(argv: Sequence[str] | None = None) -> None:
    """Delegate to :func:`sanity_check.cli.main`.

    Parameters
    ----------
    argv:
        Optional sequence of command-line arguments.  When ``None`` the
        values from :data:`sys.argv` are used, matching ``python -m``
        semantics.
    """

    if argv is None:
        argv = sys.argv[1:]
    cli.main(list(argv))


if __name__ == "__main__":  # pragma: no cover - exercised in production
    main()
