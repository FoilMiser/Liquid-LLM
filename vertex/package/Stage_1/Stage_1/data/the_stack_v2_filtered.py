"""The Stack v2 filtered code slice loader."""

from __future__ import annotations

import json
from typing import Dict, Iterator

from ..utils import open_sharded_file, resolve_glob_paths


def dataset_info() -> Dict[str, str]:
    return {
        "name": "the_stack_v2_filtered",
        "license": "SPDX permissive",
        "source": "https://huggingface.co/datasets/bigcode/the-stack-v2",
    }


def iter_samples(pattern: str) -> Iterator[dict]:
    while True:
        shards = resolve_glob_paths(pattern)
        if not shards:
            raise FileNotFoundError(f"No Stack v2 shards match pattern: {pattern}")
        for shard in shards:
            with open_sharded_file(shard, "r") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    text = record.get("text") or record.get("content") or record.get("code")
                    if not text:
                        continue
                    yield {"text": text, "source": "stack-v2-code", "kind": "code"}


__all__ = ["dataset_info", "iter_samples"]
