import gzip
import json
from pathlib import Path

from preprocess_toolkit.pipelines import c4, dolma, gsm8k


def _read_output(shard_paths):
    lines = []
    for path in shard_paths:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                lines.append(json.loads(line))
    return lines


def test_c4_pipeline(tmp_path):
    input_path = tmp_path / "c4.json.gz"
    with gzip.open(input_path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"text": "This is a sample document for testing."}) + "\n")
        fh.write(json.dumps({"text": "Another example text for C4."}) + "\n")
    summary = c4.run([("gs://bucket/c4", str(input_path))], str(tmp_path / "out"), dataset_type="lm", max_records=1, work_dir=str(tmp_path / "work"))
    assert summary["records"] >= 2
    outputs = _read_output(summary["shards"])
    assert all(sample["text"] for sample in outputs)


def test_dolma_pipeline(tmp_path):
    input_path = tmp_path / "dolma.json.gz"
    with gzip.open(input_path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"text": "Dolma text example."}) + "\n")
    summary = dolma.run([("gs://bucket/dolma", str(input_path))], str(tmp_path / "out_d"), dataset_type="lm", max_records=10, work_dir=str(tmp_path / "work_d"))
    assert summary["records"] >= 1
    outputs = _read_output(summary["shards"])
    assert outputs[0]["type"] == "lm"


def test_gsm8k_pipeline(tmp_path):
    input_path = tmp_path / "gsm8k.jsonl"
    with open(input_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"question": "What is 1+1?", "answer": "2"}) + "\n")
    summary = gsm8k.run([("gs://bucket/gsm8k", str(input_path))], str(tmp_path / "out_g"), dataset_type="math_tool", max_records=10, work_dir=str(tmp_path / "work_g"))
    assert summary["records"] >= 1
    outputs = _read_output(summary["shards"])
    assert outputs[0]["type"] == "math_tool"
    assert "Answer:" in outputs[0]["text"]
