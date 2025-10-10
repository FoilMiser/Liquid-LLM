"""Fineweb Edu Subset dataset placeholder."""

from __future__ import annotations

from typing import Dict


def dataset_info() -> Dict[str, str]:
    return {
        "name": "fineweb_edu_subset",
        "license": "CC-BY 4.0",
        "source": "https://huggingface.co/datasets/fineweb",
    }


__all__ = ["dataset_info"]
