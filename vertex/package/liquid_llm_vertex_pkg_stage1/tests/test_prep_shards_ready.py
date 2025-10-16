"""Tests for shard readiness detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1 import gcs_io, prep


def test_shards_ready_local(tmp_path):
    shard = tmp_path / "part-0000.jsonl"
    shard.write_text("{}\n", encoding="utf-8")
    assert prep.shards_ready(str(tmp_path))


def test_shards_ready_gcsfs_fallback(monkeypatch):
    class FakeFS:
        def __init__(self) -> None:
            self.calls = []

        def glob(self, pattern: str):
            self.calls.append(pattern)
            return ["bucket/data/part-0000.jsonl"]

    fake_fs = FakeFS()
    monkeypatch.setattr(gcs_io, "_has_gcloud", lambda: False)
    monkeypatch.setattr(gcs_io, "_gcsfs_filesystem", lambda: fake_fs)
    monkeypatch.setattr(prep, "list_gcs", gcs_io.list_gcs)

    assert prep.shards_ready("gs://bucket/data")
    assert fake_fs.calls
    assert fake_fs.calls[0].endswith("*.jsonl")
