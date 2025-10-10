"""The Stack V2 Filtered dataset placeholder."""

from __future__ import annotations

from typing import Dict


def dataset_info() -> Dict[str, str]:
    return {
        "name": "the_stack_v2_filtered",
        "license": "SPDX permissive",
        "source": "https://huggingface.co/datasets/bigcode/the-stack-v2",
    }


__all__ = ["dataset_info"]
