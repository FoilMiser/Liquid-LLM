"""Dolma open subset loader."""

from __future__ import annotations

import json
from typing import Dict, Iterator

from ..utils import open_sharded_file, resolve_glob_paths


def dataset_info() -> Dict[str, str]:
    return {
        "name": "dolma_subset",
        "license": "Various permissive",
        "source": "https://huggingface.co/datasets/allenai/dolma",
    }


def iter_samples(pattern: str) -> Iterator[dict]:
    while True:
        shards = resolve_glob_paths(pattern)
        if not shards:
            raise FileNotFoundError(f"No Dolma shards match pattern: {pattern}")
        for shard in shards:
            with open_sharded_file(shard, "r") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    text = (
                        record.get("text")
                        or record.get("content")
                        or record.get("body")
                        or record.get("document")
                    )
                    if not text:
                        continue
                    yield {"text": text, "source": "dolma-open", "kind": "lm"}


__all__ = ["dataset_info", "iter_samples"]
