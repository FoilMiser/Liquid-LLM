"""Dataset loading utilities for Stage-1 training."""
from __future__ import annotations

import glob
import gzip
import hashlib
import io
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - handled at runtime
    pq = None

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from . import tool_use
from .gcs_io import GCSIOError, gcs_to_local, list_gcs
from .prep import DatasetSpec, shards_ready
from .utils import configure_logging

logger = configure_logging()

_CACHE_DIR = Path("/tmp/manifest_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ManifestEntry:
    """Represents a manifest line resolved for loading."""

    path: str
    resolved_path: str
    type: str
    weight: float = 1.0
    dataset_id: Optional[str] = None


def read_manifest(
    manifest_path: str,
    datasets_cfg: Optional[Dict[str, DatasetSpec]] = None,
) -> Tuple[List[ManifestEntry], List[Dict[str, object]]]:
    logger.info("Loading manifest from %s", manifest_path)
    if manifest_path.startswith("gs://"):
        local_path = gcs_to_local(manifest_path, "/tmp/manifest.jsonl")
    else:
        local_path = manifest_path
    entries: List[ManifestEntry] = []
    snapshot: List[Dict[str, object]] = []
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            raw_path = obj["path"]
            dtype = obj.get("type", "lm")
            weight = float(obj.get("weight", 1.0))
            dataset_id = _match_dataset(raw_path, datasets_cfg)
            resolved_path = _resolve_dataset_path(raw_path, dataset_id, datasets_cfg)
            entries.append(
                ManifestEntry(
                    path=raw_path,
                    resolved_path=resolved_path,
                    type=dtype,
                    weight=weight,
                    dataset_id=dataset_id,
                )
            )
            snapshot.append(
                {
                    "dataset_id": dataset_id,
                    "type": dtype,
                    "weight": weight,
                    "path": raw_path,
                    "resolved_path": resolved_path,
                }
            )
    return entries, snapshot


def _match_dataset(path: str, datasets_cfg: Optional[Dict[str, DatasetSpec]]) -> Optional[str]:
    if not datasets_cfg:
        return None
    normalized = path.rstrip("/")
    for job, spec in datasets_cfg.items():
        for candidate in filter(None, (spec.manifest, spec.inp, spec.out)):
            candidate_norm = candidate.rstrip("/")
            if normalized == candidate_norm or normalized.startswith(candidate_norm):
                return job
    return None


def _resolve_dataset_path(
    raw_path: str,
    dataset_id: Optional[str],
    datasets_cfg: Optional[Dict[str, DatasetSpec]],
) -> str:
    if not datasets_cfg or not dataset_id:
        return raw_path
    spec = datasets_cfg.get(dataset_id)
    if not spec:
        return raw_path
    extension = Path(raw_path).suffix
    if extension in {".parquet", ".gz"} and shards_ready(spec.out):
        return spec.shard_glob()
    return raw_path


def expand_gcs_pattern(pattern: str) -> List[str]:
    if pattern.startswith("gs://"):
        try:
            return list_gcs(pattern)
        except GCSIOError:
            logger.warning("Failed to expand GCS pattern %s", pattern)
            return [pattern]
    matched = glob.glob(pattern)
    return matched if matched else [pattern]


def _resolve_local(path: str) -> str:
    if path.startswith("gs://"):
        digest = hashlib.md5(path.encode("utf-8")).hexdigest()
        local = _CACHE_DIR / f"{digest}_{Path(path).name}"
        if not local.exists():
            gcs_to_local(path, str(local))
        return str(local)
    return path


def _stream_json_lines(local_path: str) -> Iterator[Dict[str, object]]:
    with open(local_path, "rb") as fh:
        reader = io.BufferedReader(fh)
        for raw in reader:
            line = raw.decode("utf-8")
            if not line.strip():
                continue
            yield json.loads(line)


def _stream_json_gz(local_path: str) -> Iterator[Dict[str, object]]:
    with gzip.open(local_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            yield json.loads(line)


def _stream_parquet(local_path: str) -> Iterator[Dict[str, object]]:
    if pq is None:
        raise RuntimeError("pyarrow is required to read parquet manifests")
    table = pq.ParquetFile(local_path)
    for batch in table.iter_batches():  # pragma: no cover - requires pyarrow
        for row in batch.to_pylist():
            yield row


def load_records(path: str) -> Iterator[Dict[str, object]]:
    local = _resolve_local(path)
    suffix = Path(local).suffix
    if suffix == ".gz":
        yield from _stream_json_gz(local)
    elif suffix == ".parquet":
        yield from _stream_parquet(local)
    else:
        yield from _stream_json_lines(local)


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
        self.dataset_counts: Dict[str, int] = {}
        self._build_index()

    def _build_index(self) -> None:
        running_id = 0
        counts: Dict[str, int] = defaultdict(int)
        for entry in self.entries:
            resolved_paths = expand_gcs_pattern(entry.resolved_path)
            dataset_key = entry.dataset_id or entry.resolved_path
            for path in resolved_paths:
                for obj in load_records(path):
                    text = obj.get("text", "")
                    if not isinstance(text, str) or not text:
                        continue
                    sample_type = obj.get("type", entry.type)
                    if sample_type == "math_tool":
                        text = tool_use.traces.maybe_inject_tool_result(text)
                    sample_id = obj.get("sample_id") or f"{dataset_key}_sample_{running_id}"
                    payload = {
                        "text": text,
                        "type": sample_type,
                        "sample_id": sample_id,
                    }
                    self.samples.append(payload)
                    counts[dataset_key] += 1
                    running_id += 1
        self.dataset_counts = dict(counts)
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
