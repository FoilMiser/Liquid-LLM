"""ASDiv tool-trace SFT loader."""

from __future__ import annotations

import json
from typing import Dict, Iterator, List

from ..utils import open_sharded_file, resolve_glob_paths


def dataset_info() -> Dict[str, str]:
    return {
        "name": "asdiv_sft",
        "license": "CC BY 4.0",
        "source": "https://github.com/chaochun/nlp-architect",
    }


def _format_trace(trace) -> List[str]:
    if trace is None:
        return []
    if isinstance(trace, list):
        return [str(item).strip() for item in trace if str(item).strip()]
    if isinstance(trace, dict):
        return [f"{key}: {value}" for key, value in trace.items()]
    text = str(trace).strip()
    return [text] if text else []


def iter_samples(pattern: str) -> Iterator[dict]:
    while True:
        shards = resolve_glob_paths(pattern)
        if not shards:
            raise FileNotFoundError(f"No ASDiv tool shards match pattern: {pattern}")
        for shard in shards:
            with open_sharded_file(shard, "r") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    prompt = record.get("prompt") or record.get("question") or record.get("input")
                    answer = record.get("answer") or record.get("completion") or record.get("output")
                    trace_lines = _format_trace(record.get("tool_trace") or record.get("trace") or record.get("tool_calls"))
                    segments = [prompt] + trace_lines + [answer]
                    text = "\n".join(str(seg).strip() for seg in segments if seg)
                    if not text:
                        continue
                    yield {"text": text, "source": "asdiv-tool", "kind": "math_tool"}


__all__ = ["dataset_info", "iter_samples"]
