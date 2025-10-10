"""Asdiv Sft dataset placeholder."""

from __future__ import annotations

from typing import Dict


def dataset_info() -> Dict[str, str]:
    return {
        "name": "asdiv_sft",
        "license": "CC BY 4.0",
        "source": "https://github.com/chaochun/nlp-architect",
    }


__all__ = ["dataset_info"]
