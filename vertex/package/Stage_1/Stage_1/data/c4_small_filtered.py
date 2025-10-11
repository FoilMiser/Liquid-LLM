"""C4 filtered slice loader."""

from __future__ import annotations

import json
from typing import Dict, Iterator

from ..utils import open_sharded_file, resolve_glob_paths


def dataset_info() -> Dict[str, str]:
    return {
        "name": "c4_small_filtered",
        "license": "Apache-2.0",
        "source": "https://www.tensorflow.org/datasets/catalog/c4",
    }


def iter_samples(pattern: str) -> Iterator[dict]:
    while True:
        shards = resolve_glob_paths(pattern)
        if not shards:
            raise FileNotFoundError(f"No C4 shards match pattern: {pattern}")
        for shard in shards:
            with open_sharded_file(shard, "r") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    text = record.get("text") or record.get("content")
                    if not text:
                        continue
                    yield {"text": text, "source": "c4-filtered-small", "kind": "lm"}


__all__ = ["dataset_info", "iter_samples"]
