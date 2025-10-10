"""Weighted dataset mixer for Stage-1."""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional

from .manifest_parser import ManifestEntry


@dataclass
class DatasetSpec:
    path: str
    weight: float
    kind: str


class DataMixer:
    def __init__(self, manifest: Iterable[ManifestEntry], tool_ratio: float = 0.08, seed: int = 42):
        entries = list(manifest)
        self.datasets: List[DatasetSpec] = [DatasetSpec(e.path, e.weight, e.type) for e in entries]
        total = sum(spec.weight for spec in self.datasets)
        for spec in self.datasets:
            spec.weight = spec.weight / total if total > 0 else 0
        self.tool_ratio = tool_ratio
        self.random = random.Random(seed)

    def summary(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for spec in self.datasets:
            totals[spec.kind] = totals.get(spec.kind, 0.0) + spec.weight
        return totals

    def iter_order(self) -> Iterator[DatasetSpec]:
        tool_specs = [spec for spec in self.datasets if "tool" in spec.kind]
        non_tool_specs = [spec for spec in self.datasets if "tool" not in spec.kind]
        while True:
            if tool_specs and self.random.random() < self.tool_ratio:
                yield self.random.choices(tool_specs, weights=[s.weight for s in tool_specs])[0]
            elif non_tool_specs:
                yield self.random.choices(non_tool_specs, weights=[s.weight for s in non_tool_specs])[0]
            else:
                yield self.random.choice(tool_specs)

    def iter_samples(self) -> Iterator[Dict[str, str]]:
        for spec in self.iter_order():
            yield {"source": spec.path, "type": spec.kind}


__all__ = ["DataMixer", "DatasetSpec"]
