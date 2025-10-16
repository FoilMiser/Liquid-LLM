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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from . import tool_use
from .gcs_io import GCSIOError, gcs_to_local, list_gcs
from .prep import DatasetSpec, shards_ready
from .utils import configure_logging

logger = configure_logging()

_CACHE_DIR = Path("/tmp/manifest_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_LOGIT_CACHE_DIR = Path("/tmp/teacher_logits")
_LOGIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
    if shards_ready(spec.out):
        return spec.shard_glob()
    if extension in {".jsonl", ".json", ".json.gz", ".parquet"}:
        return raw_path
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
        teacher_mode: str = "precompute",
        teacher_logits_dir: Optional[str] = None,
        *,
        hybrid: bool = False,
        math_logits_dir: Optional[str] = None,
        code_logits_dir: Optional[str] = None,
        hybrid_strict: bool = False,
        split: str = "train",
    ) -> None:
        self.entries = list(manifest_entries)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tool_use_ratio = tool_use_ratio
        self.samples: List[Dict[str, str]] = []
        self.dataset_counts: Dict[str, int] = {}
        self.teacher_mode = teacher_mode
        self.teacher_logits_dir = teacher_logits_dir.rstrip("/") if teacher_logits_dir else None
        self.hybrid = hybrid
        self.math_logits_dir = math_logits_dir.rstrip("/") if math_logits_dir else None
        self.code_logits_dir = code_logits_dir.rstrip("/") if code_logits_dir else None
        self.hybrid_strict = hybrid_strict
        self.split = split
        self._teacher_cache: Dict[tuple[str, str], torch.Tensor] = {}
        self._auto_sample_counts: Dict[str, int] = defaultdict(int)
        self._bad_shape_warned: set[str] = set()
        self._missing_warned: Dict[str, set[str]] = defaultdict(set)
        self._epoch_total = 0
        self._epoch_missing = 0
        self.sample_type_counts: Dict[str, int] = {}
        self._hybrid_missing_warned_types: set[str] = set()
        self._build_index()

    def _build_index(self) -> None:
        counts: Dict[str, int] = defaultdict(int)
        type_counts: Dict[str, int] = defaultdict(int)
        for entry in self.entries:
            resolved_paths = expand_gcs_pattern(entry.resolved_path)
            dataset_key = entry.dataset_id or entry.resolved_path
            for path in resolved_paths:
                for row_idx, obj in enumerate(load_records(path)):
                    text = obj.get("text", "")
                    if not isinstance(text, str) or not text:
                        continue
                    sample_type = obj.get("type", entry.type)
                    if sample_type == "math_tool":
                        text = tool_use.traces.maybe_inject_tool_result(text)
                    sample_id = obj.get("sample_id")
                    if not sample_id:
                        digest = hashlib.sha1(
                            f"{path}:{row_idx}:{text[:200]}".encode("utf-8")
                        ).hexdigest()
                        sample_id = f"auto_{digest}"
                        self._auto_sample_counts[dataset_key] += 1
                    payload = {
                        "text": text,
                        "sample_type": sample_type,
                        "sample_id": sample_id,
                    }
                    self.samples.append(payload)
                    counts[dataset_key] += 1
                    type_counts[sample_type] += 1
        self.dataset_counts = dict(counts)
        self.auto_sample_counts = dict(self._auto_sample_counts)
        self.sample_type_counts = dict(type_counts)
        logger.info("Loaded %d samples from manifest", len(self.samples))
        for dataset_key, missing in self._auto_sample_counts.items():
            if missing:
                logger.warning(
                    "Dataset %s missing sample_id for %d records; auto-generated using SHA1",
                    dataset_key,
                    missing,
                )

    def begin_epoch(self) -> None:
        self._epoch_total = 0
        self._epoch_missing = 0

    def epoch_teacher_stats(self) -> tuple[int, int]:
        return self._epoch_total, self._epoch_missing

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        tokens = self.tokenizer(
            sample["text"],
            truncation=True,
            max_length=self.seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        sample_type = sample["sample_type"]
        item: Dict[str, object] = {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "sample_type": sample_type,
            "sample_id": sample["sample_id"],
        }
        item["type"] = sample_type
        directory: Optional[str] = None
        source_tag = "disabled"
        if self.teacher_mode == "precompute" and self.teacher_logits_dir:
            directory = self.teacher_logits_dir
            source_tag = "precompute"
        elif self.hybrid and sample_type in {"math_tool", "code"}:
            if sample_type == "math_tool":
                directory = self.math_logits_dir
            else:
                directory = self.code_logits_dir
            source_tag = sample_type
        if directory:
            logits, status = self._load_teacher_logits(sample["sample_id"], directory, source_tag)
            self._epoch_total += 1
            if status in {"missing", "invalid"}:
                self._epoch_missing += 1
            if logits is not None:
                item["teacher_logits"] = logits
            if status == "missing" and self.hybrid and self.hybrid_strict and source_tag in {"math_tool", "code"}:
                warn_key = f"{self.split}:{source_tag}"
                if warn_key not in self._hybrid_missing_warned_types:
                    self._hybrid_missing_warned_types.add(warn_key)
                    logger.warning(
                        "Hybrid strict mode: missing %s logits detected on split=%s; disabling KD for affected batches",
                        source_tag,
                        self.split,
                    )
                item["hybrid_skip_kd"] = True
            item["teacher_status"] = status
        return item

    def _materialize_teacher_path(self, directory: str, sample_id: str, ext: str) -> Optional[str]:
        base = f"{directory}/{sample_id}{ext}"
        if base.startswith("gs://"):
            cache_name = hashlib.md5(base.encode("utf-8")).hexdigest()
            local_path = _LOGIT_CACHE_DIR / f"{cache_name}{ext}"
            if local_path.exists():
                return str(local_path)
            try:
                return gcs_to_local(base, str(local_path))
            except GCSIOError:
                return None
        path = Path(base)
        if path.exists():
            return str(path)
        return None

    def _normalize_teacher_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if logits.dim() != 2:
            logits = logits.view(-1, logits.shape[-1])
        seq_len, vocab_size = logits.shape
        if seq_len < self.seq_len:
            logits = F.pad(logits, (0, 0, 0, self.seq_len - seq_len))
        elif seq_len > self.seq_len:
            logits = logits[: self.seq_len]
        target_vocab = getattr(self.tokenizer, "vocab_size", vocab_size)
        if vocab_size < target_vocab:
            logits = F.pad(logits, (0, target_vocab - vocab_size))
        elif vocab_size > target_vocab:
            logits = logits[:, :target_vocab]
        return logits

def _load_teacher_logits(
    self, sample_id: str, directory: str, tag: str
) -> tuple[Optional[torch.Tensor], str]:
        cache_key = (directory, sample_id)
        if cache_key in self._teacher_cache:
            return self._teacher_cache[cache_key], "cached"
        if not directory:
            return None, "disabled"
        for ext in (".pt", ".npy"):
            path = self._materialize_teacher_path(directory, sample_id, ext)
            if not path:
                continue
            try:
                if ext == ".pt":
                    tensor = torch.load(path, map_location="cpu")
                else:
                    tensor = torch.from_numpy(np.load(path, allow_pickle=False))
            except Exception:
                continue
            logits = torch.as_tensor(tensor)
            if logits.dim() != 2:
                key = f"shape:{Path(path).suffix}:{tag}"
                if key not in self._bad_shape_warned:
                    self._bad_shape_warned.add(key)
                    logger.warning(
                        "Teacher logits at %s have invalid shape %s; expected [T, V]; skipping",
                        path,
                        tuple(logits.shape),
                    )
                return None, "invalid"
            logits = self._normalize_teacher_logits(logits)
            self._teacher_cache[cache_key] = logits
            return logits, "ok"
        if sample_id not in self._missing_warned[directory]:
            self._missing_warned[directory].add(sample_id)
            logger.warning("Teacher logits missing for sample %s in %s", sample_id, directory)
        return None, "missing"


def collate_batch(samples: Sequence[Dict[str, object]]) -> Dict[str, object]:
    input_ids = torch.stack([sample["input_ids"] for sample in samples])
    attention_mask = torch.stack([sample["attention_mask"] for sample in samples])
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    sample_types = [str(sample.get("sample_type", "lm")) for sample in samples]
    teacher_logits = [sample.get("teacher_logits") for sample in samples]
    teacher_status = [sample.get("teacher_status", "disabled") for sample in samples]
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "sample_ids": sample_ids,
        "sample_types": sample_types,
        "teacher_logits": teacher_logits,
        "teacher_status": teacher_status,
    }
    if any(sample.get("hybrid_skip_kd") for sample in samples):
        batch["hybrid_skip_kd"] = True
    return batch


def build_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
