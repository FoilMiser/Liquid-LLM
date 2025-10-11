"""GSM8K tool-trace SFT loader."""

from __future__ import annotations

import json
from typing import Dict, Iterator, List

from ..utils import open_sharded_file, resolve_glob_paths


def dataset_info() -> Dict[str, str]:
    return {
        "name": "gsm8k_sft",
        "license": "MIT",
        "source": "https://github.com/openai/grade-school-math",
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
            raise FileNotFoundError(f"No GSM8K tool shards match pattern: {pattern}")
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
                    yield {"text": text, "source": "gsm8k-tool", "kind": "math_tool"}


__all__ = ["dataset_info", "iter_samples"]
