import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1 import data


class _FakeTokenizer:
    def __init__(self, vocab_size: int = 16, seq_len: int = 8) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.pad_token = 0

    def __call__(self, text, truncation, max_length, padding, return_tensors):
        tokens = torch.zeros(max_length, dtype=torch.long)
        mask = torch.zeros(max_length, dtype=torch.long)
        length = min(len(text), max_length)
        if length > 0:
            tokens[:length] = torch.arange(length) % self.vocab_size
            mask[:length] = 1
        return {"input_ids": tokens.unsqueeze(0), "attention_mask": mask.unsqueeze(0)}


def test_precompute_loader_includes_teacher_logits(tmp_path):
    manifest_path = tmp_path / "data.jsonl"
    sample = {"text": "hello world", "type": "lm", "sample_id": "sample123"}
    manifest_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    entry = data.ManifestEntry(path=str(manifest_path), resolved_path=str(manifest_path), type="lm")
    tokenizer = _FakeTokenizer()
    logits_dir = tmp_path / "logits"
    logits_dir.mkdir()
    torch.save(torch.randn(tokenizer.seq_len, tokenizer.vocab_size), logits_dir / "sample123.pt")

    dataset = data.ManifestDataset(
        [entry],
        tokenizer,
        seq_len=tokenizer.seq_len,
        tool_use_ratio=0.0,
        teacher_mode="precompute",
        teacher_logits_dir=str(logits_dir),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    assert "teacher_logits" in batch
    assert batch["teacher_logits"].shape[-2:] == (tokenizer.seq_len, tokenizer.vocab_size)
    assert batch["teacher_status"][0] == "ok"
