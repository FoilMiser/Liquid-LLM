"""Dolma Subset dataset placeholder."""

from __future__ import annotations

from typing import Dict


def dataset_info() -> Dict[str, str]:
    return {
        "name": "dolma_subset",
        "license": "Various permissive",
        "source": "https://huggingface.co/datasets/allenai/dolma",
    }


__all__ = ["dataset_info"]
