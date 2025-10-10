"""WikiText-103 dataset loader (CC BY-SA 3.0)."""

from __future__ import annotations

from typing import Dict


def dataset_info() -> Dict[str, str]:
    return {
        "name": "WikiText-103",
        "license": "CC BY-SA 3.0",
        "source": "https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/",
    }


__all__ = ["dataset_info"]
