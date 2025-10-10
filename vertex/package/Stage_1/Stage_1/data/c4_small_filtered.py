"""C4 Small Filtered dataset placeholder."""

from __future__ import annotations

from typing import Dict


def dataset_info() -> Dict[str, str]:
    return {
        "name": "c4_small_filtered",
        "license": "Apache-2.0",
        "source": "https://www.tensorflow.org/datasets/catalog/c4",
    }


__all__ = ["dataset_info"]
