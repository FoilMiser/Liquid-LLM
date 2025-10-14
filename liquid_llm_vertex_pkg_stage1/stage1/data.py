"""Data loading utilities for Stage 1 training."""

import json
import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List

if TYPE_CHECKING:  # pragma: no cover
    import gcsfs

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerBase

from .tool_use.traces import maybe_inject_tool_result

LOGGER = logging.getLogger(__name__)


@dataclass
class ManifestEntry:
    path: str
    type: str
    weight: float


def load_manifest(manifest_uri: str) -> List[ManifestEntry]:
    import gcsfs

    fs = gcsfs.GCSFileSystem()
    entries: List[ManifestEntry] = []
    LOGGER.info("Loading manifest from %s", manifest_uri)
    with fs.open(manifest_uri, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            entries.append(
                ManifestEntry(
                    path=payload["path"],
                    type=payload.get("type", "lm"),
                    weight=float(payload.get("weight", 1.0)),
                )
            )
    return entries


def resolve_paths(fs: "gcsfs.GCSFileSystem", pattern: str) -> List[str]:
    if any(token in pattern for token in "*?[]"):
        return fs.glob(pattern)
    return [pattern]


def iter_jsonl(fs: "gcsfs.GCSFileSystem", gcs_path: str) -> Iterator[Dict[str, str]]:
    with fs.open(gcs_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def tokenize_and_pack(
    tokenizer: PreTrainedTokenizerBase,
    texts: Iterable[str],
    block_size: int,
) -> Iterator[Dict[str, torch.Tensor]]:
    buffer: List[int] = []
    eos_id = tokenizer.eos_token_id
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if eos_id is not None:
            ids = ids + [eos_id]
        buffer.extend(ids)
        while len(buffer) >= block_size + 1:
            input_ids = buffer[:block_size]
            labels = buffer[1 : block_size + 1]
            buffer = buffer[block_size:]
            attention_mask = [1] * block_size
            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }


class PackedDataset(IterableDataset):
    def __init__(
        self,
        entries: List[ManifestEntry],
        tokenizer: PreTrainedTokenizerBase,
        block_size: int,
        seed: int = 42,
        tool_use_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.entries = entries
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.seed = seed
        self.tool_use_ratio = tool_use_ratio
        import gcsfs

        self.fs = gcsfs.GCSFileSystem()

    def _stream_entry(self, entry: ManifestEntry) -> Iterator[Dict[str, torch.Tensor]]:
        paths = resolve_paths(self.fs, entry.path)
        rng = random.Random(self.seed)
        rng.shuffle(paths)

        def text_iter() -> Iterator[str]:
            for path in paths:
                for row in iter_jsonl(self.fs, path):
                    text = row.get("text", "")
                    if entry.type == "math_tool" and rng.random() <= self.tool_use_ratio:
                        text = maybe_inject_tool_result(text)
                    yield text

        return tokenize_and_pack(self.tokenizer, text_iter(), self.block_size)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self.seed)
        iterators: Dict[str, Iterator[Dict[str, torch.Tensor]]] = {}
        weights = [entry.weight for entry in self.entries]
        if not weights:
            return iter([])
        while True:
            entry = rng.choices(self.entries, weights=weights, k=1)[0]
            key = entry.path
            if key not in iterators:
                iterators[key] = self._stream_entry(entry)
            try:
                yield next(iterators[key])
            except StopIteration:
                iterators[key] = self._stream_entry(entry)
                yield next(iterators[key])


class DataModule:
    def __init__(
        self,
        manifest_uri: str,
        tokenizer: PreTrainedTokenizerBase,
        block_size: int,
        batch_size: int,
        seed: int = 42,
        tool_use_ratio: float = 0.0,
    ) -> None:
        self.entries = load_manifest(manifest_uri)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.seed = seed
        self.tool_use_ratio = tool_use_ratio

    def train_dataloader(self) -> DataLoader:
        dataset = PackedDataset(
            self.entries,
            tokenizer=self.tokenizer,
            block_size=self.block_size,
            seed=self.seed,
            tool_use_ratio=self.tool_use_ratio,
        )
        return DataLoader(dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        dataset = PackedDataset(
            self.entries[: max(1, len(self.entries) // 10)],
            tokenizer=self.tokenizer,
            block_size=self.block_size,
            seed=self.seed + 1,
            tool_use_ratio=self.tool_use_ratio,
        )
        return DataLoader(dataset, batch_size=self.batch_size)


__all__ = [
    "DataModule",
    "load_manifest",
    "ManifestEntry",
    "resolve_paths",
    "iter_jsonl",
    "tokenize_and_pack",
]
