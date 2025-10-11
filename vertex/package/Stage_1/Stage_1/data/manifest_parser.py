"""Manifest parser for dataset configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from ..utils import open_sharded_file


@dataclass
class ManifestEntry:
    path: str
    type: str
    weight: float
    dataset: str | None = None


def load_manifest(path: str) -> List[ManifestEntry]:
    entries: List[ManifestEntry] = []
    with open_sharded_file(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            entries.append(
                ManifestEntry(
                    path=data["path"],
                    type=data.get("type", "lm"),
                    weight=float(data["weight"]),
                    dataset=data.get("dataset"),
                )
            )
    return entries


__all__ = ["ManifestEntry", "load_manifest"]
