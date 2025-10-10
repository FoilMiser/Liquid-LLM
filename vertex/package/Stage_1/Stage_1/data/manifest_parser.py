"""Manifest parser for dataset configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class ManifestEntry:
    path: str
    type: str
    weight: float


def load_manifest(path: str) -> List[ManifestEntry]:
    entries: List[ManifestEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            entries.append(ManifestEntry(path=data["path"], type=data["type"], weight=float(data["weight"])))
    return entries


__all__ = ["ManifestEntry", "load_manifest"]
