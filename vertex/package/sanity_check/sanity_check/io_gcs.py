"""Helpers for checkpoint IO from Google Cloud Storage."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Optional

import fsspec


def is_gcs_uri(uri: str) -> bool:
    return uri.startswith("gs://")


def download_checkpoint(uri: str, target_dir: Path, *, retries: int = 3, delay: float = 2.0) -> Path:
    """Download a checkpoint from GCS to the target directory."""

    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / "checkpoint.pt"

    if not is_gcs_uri(uri):
        source = Path(uri)
        if not source.exists():
            raise FileNotFoundError(f"Checkpoint not found at {source}")
        return source

    fs = fsspec.filesystem("gcs")
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            with fs.open(uri, "rb") as src, open(destination, "wb") as dst:
                while True:
                    chunk = src.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            return destination
        except Exception as exc:  # pragma: no cover - difficult to simulate
            last_error = exc
            if attempt == retries:
                raise
            time.sleep(delay * attempt)
    if last_error:
        raise last_error
    raise RuntimeError("Failed to download checkpoint due to unknown error")


def hash_state_dict(state_dict: dict) -> str:
    """Compute a stable hash of tensor shapes and dtypes."""

    digest = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        shape = ",".join(str(dim) for dim in tensor.shape)
        digest.update(name.encode("utf-8"))
        digest.update(shape.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
    return digest.hexdigest()
