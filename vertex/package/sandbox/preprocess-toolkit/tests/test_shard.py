import json
from pathlib import Path

from preprocess_toolkit import shard


def test_shard_rotation_and_schema(tmp_path):
    writer = shard.ShardWriter(tmp_path, "lm", max_records=2)
    texts = ["Sample one text", "Sample two text", "Sample three text"]
    for idx, text in enumerate(texts):
        sample_id = shard.compute_sample_id("gs://bucket/file", idx, text)
        writer.write(sample_id, text)
    writer.close()

    shard_files = sorted(Path(tmp_path).glob("*.jsonl"))
    assert len(shard_files) == 2
    payloads = []
    for path in shard_files:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                payload = json.loads(line)
                assert set(payload) == {"sample_id", "type", "text"}
                payloads.append(payload)
    assert len(payloads) == 3
    assert payloads[0]["sample_id"] == shard.compute_sample_id("gs://bucket/file", 0, texts[0])


def test_dedup_within_shard(tmp_path):
    writer = shard.ShardWriter(tmp_path, "lm", max_records=10)
    text = "Duplicate text"
    sample_id = shard.compute_sample_id("gs://bucket/file", 0, text)
    assert writer.write(sample_id, text) is True
    sample_id2 = shard.compute_sample_id("gs://bucket/file", 1, text)
    assert writer.write(sample_id2, text) is False
    writer.close()
