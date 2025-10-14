"""Dataset loading utilities for Stage-1 training."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from . import tool_use
from .gcs_io import gcs_to_local, list_gcs
from .utils import configure_logging

logger = configure_logging()


@dataclass
class ManifestEntry:
    path: str
    type: str
    weight: float = 1.0


def read_manifest(manifest_path: str) -> List[ManifestEntry]:
    logger.info("Loading manifest from %s", manifest_path)
    if manifest_path.startswith("gs://"):
        local_path = gcs_to_local(manifest_path, "/tmp/manifest.jsonl")
    else:
        local_path = manifest_path
    entries: List[ManifestEntry] = []
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            entries.append(ManifestEntry(path=obj["path"], type=obj.get("type", "lm"), weight=float(obj.get("weight", 1.0))))
    return entries


def expand_gcs_pattern(pattern: str) -> List[str]:
    if not pattern.startswith("gs://"):
        return [pattern]
    try:
        return list_gcs(pattern)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to expand pattern %s: %s", pattern, exc)
        return [pattern]


def load_jsonl(path: str) -> Iterator[Dict[str, str]]:
    if path.startswith("gs://"):
        local = gcs_to_local(path, "/tmp/dataset.jsonl")
    else:
        local = path
    with open(local, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


class ManifestDataset(Dataset):
    """Torch dataset built from a manifest file."""

    def __init__(
        self,
        manifest_entries: Sequence[ManifestEntry],
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
        tool_use_ratio: float = 0.0,
    ) -> None:
        self.entries = list(manifest_entries)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tool_use_ratio = tool_use_ratio
        self.samples: List[Dict[str, str]] = []
        self._build_index()

    def _build_index(self) -> None:
        running_id = 0
        for entry in self.entries:
            for path in expand_gcs_pattern(entry.path):
                for obj in load_jsonl(path):
                    sample_type = obj.get("type", entry.type)
                    text = obj.get("text", "")
                    if not text:
                        continue
                    if sample_type == "math_tool":
                        text = tool_use.traces.maybe_inject_tool_result(text)
                    self.samples.append({"text": text, "type": sample_type, "sample_id": f"sample_{running_id}"})
                    running_id += 1
        logger.info("Loaded %d samples from manifest", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        tokens = self.tokenizer(
            sample["text"],
            truncation=True,
            max_length=self.seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "type": sample["type"],
            "sample_id": sample["sample_id"],
        }


def build_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
