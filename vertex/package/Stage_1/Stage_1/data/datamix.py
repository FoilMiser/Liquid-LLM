"""Weighted dataset mixer for Stage-1."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

from . import (
    asdiv_sft,
    c4_small_filtered,
    dolma_subset,
    fineweb_edu_subset,
    gsm8k_sft,
    svamp_sft,
    the_stack_v2_filtered,
    wikitext,
    wikipedia,
)
from .manifest_parser import ManifestEntry


DATASET_REGISTRY = {
    "wikitext-103": wikitext.iter_samples,
    "wikipedia-clean": wikipedia.iter_samples,
    "c4-filtered-small": c4_small_filtered.iter_samples,
    "fineweb-edu": fineweb_edu_subset.iter_samples,
    "dolma-open": dolma_subset.iter_samples,
    "gsm8k-tool": gsm8k_sft.iter_samples,
    "svamp-tool": svamp_sft.iter_samples,
    "asdiv-tool": asdiv_sft.iter_samples,
    "stack-v2-code": the_stack_v2_filtered.iter_samples,
}


@dataclass
class DatasetSpec:
    name: str
    kind: str
    path: str
    weight: float
    key: str


class DataMixer:
    def __init__(self, manifest: Iterator[ManifestEntry], tool_ratio: float = 0.08, seed: int = 42):
        entries = list(manifest)
        if not entries:
            raise ValueError("Dataset manifest must contain at least one entry")

        self.random = random.Random(seed)
        self.tool_ratio = max(0.0, min(1.0, tool_ratio))

        specs: List[DatasetSpec] = []
        iterators: List[Iterator[dict]] = []
        for entry in entries:
            dataset_name = (entry.dataset or entry.type).lower()
            loader = DATASET_REGISTRY.get(dataset_name)
            if loader is None:
                raise ValueError(f"Unknown dataset '{dataset_name}' in manifest")
            spec = DatasetSpec(
                name=entry.dataset or entry.type,
                kind=entry.type,
                path=entry.path,
                weight=max(0.0, float(entry.weight)),
                key=dataset_name,
            )
            specs.append(spec)
            iterators.append(loader(entry.path))

        total_weight = sum(spec.weight for spec in specs)
        if total_weight <= 0:
            raise ValueError("Dataset weights must sum to a positive value")

        self._entries: List[Tuple[DatasetSpec, Iterator[dict]]] = list(zip(specs, iterators))
        self._weights: List[float] = [spec.weight / total_weight for spec in specs]

        self._tool_indices = [idx for idx, (spec, _) in enumerate(self._entries) if "tool" in spec.kind.lower()]
        self._base_indices = [idx for idx in range(len(self._entries)) if idx not in self._tool_indices]
        if not self._base_indices:
            self._base_indices = list(range(len(self._entries)))
        if not self._tool_indices:
            self._tool_indices = list(range(len(self._entries)))

    def summary(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for (spec, _), weight in zip(self._entries, self._weights):
            totals[spec.name] = totals.get(spec.name, 0.0) + weight
        return totals

    def iter_order(self) -> Iterator[int]:
        while True:
            if self._tool_indices and self.random.random() < self.tool_ratio:
                weights = [self._weights[i] for i in self._tool_indices]
                idx = self.random.choices(self._tool_indices, weights=weights, k=1)[0]
            else:
                weights = [self._weights[i] for i in self._base_indices]
                idx = self.random.choices(self._base_indices, weights=weights, k=1)[0]
            yield idx

    def iter_samples(self) -> Iterator[dict]:
        for idx in self.iter_order():
            spec, iterator = self._entries[idx]
            try:
                sample = next(iterator)
            except StopIteration:
                iterator = DATASET_REGISTRY[spec.key](spec.path)
                self._entries[idx] = (spec, iterator)
                sample = next(iterator)
            if "source" not in sample:
                sample["source"] = spec.name
            if "kind" not in sample:
                sample["kind"] = spec.kind
            yield sample


__all__ = ["DataMixer", "DatasetSpec"]
