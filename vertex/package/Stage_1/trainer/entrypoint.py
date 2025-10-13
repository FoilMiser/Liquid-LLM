"""Compatibility shim delegating to ``Stage_1.vertex.entrypoint``."""

from __future__ import annotations

from Stage_1.vertex.entrypoint import *  # noqa: F401,F403

# The public surface of ``Stage_1.vertex.entrypoint`` is re-exported so that
# existing scripts importing the legacy ``trainer`` package keep functioning
# without modifications.
