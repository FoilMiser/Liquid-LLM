"""Wikipedia dataset placeholder."""

from __future__ import annotations

from typing import Dict


def dataset_info() -> Dict[str, str]:
    return {
        "name": "wikipedia",
        "license": "CC BY-SA 3.0",
        "source": "https://dumps.wikimedia.org/",
    }


__all__ = ["dataset_info"]
