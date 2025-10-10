"""Svamp Sft dataset placeholder."""

from __future__ import annotations

from typing import Dict


def dataset_info() -> Dict[str, str]:
    return {
        "name": "svamp_sft",
        "license": "MIT",
        "source": "https://github.com/arkilpatel/SVAMP",
    }


__all__ = ["dataset_info"]
