"""Tests that manifest loading prefers shard JSONL outputs."""

import json

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1 import data
from stage1.prep import DatasetSpec


class DummyTokenizer:
    def __call__(self, text, truncation, max_length, padding, return_tensors):
        assert truncation and padding == "max_length"
        return {
            "input_ids": torch.zeros((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


def test_manifest_prefers_shards(tmp_path):
    raw_path = tmp_path / "raw.parquet"
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps({"path": str(raw_path), "type": "lm", "weight": 1.0}) + "\n", encoding="utf-8")

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shard_file = shard_dir / "part-0000.jsonl"
    shard_file.write_text(json.dumps({"text": "hello", "type": "lm", "sample_id": "s1"}) + "\n", encoding="utf-8")

    datasets_cfg = {
        "demo": DatasetSpec(job="demo", inp=str(raw_path), out=str(shard_dir), dtype="lm", manifest=None)
    }

    entries, snapshot = data.read_manifest(str(manifest_path), datasets_cfg)
    assert entries[0].dataset_id == "demo"
    assert entries[0].resolved_path.startswith(str(shard_dir))
    assert entries[0].resolved_path.endswith("*.jsonl")
    assert snapshot[0]["resolved_path"].endswith("*.jsonl")

    tokenizer = DummyTokenizer()
    dataset = data.ManifestDataset(entries, tokenizer, seq_len=8)
    assert dataset.dataset_counts.get("demo") == 1
    assert len(dataset) == 1
