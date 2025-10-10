"""Gsm8K Sft dataset placeholder."""

from __future__ import annotations

from typing import Dict


def dataset_info() -> Dict[str, str]:
    return {
        "name": "gsm8k_sft",
        "license": "MIT",
        "source": "https://github.com/openai/grade-school-math",
    }


__all__ = ["dataset_info"]
