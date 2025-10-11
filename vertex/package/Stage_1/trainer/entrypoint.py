"""Module shim so that ``python -m trainer.entrypoint`` keeps working."""

from __future__ import annotations

from Stage_1.vertex.entrypoint import *  # noqa: F401,F403

# The public surface of ``Stage_1.vertex.entrypoint`` is re-exported so that
# existing scripts importing ``trainer.entrypoint`` keep functioning without
# modifications.
