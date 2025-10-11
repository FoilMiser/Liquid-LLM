"""GCS/local IO helpers."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

import fnmatch
import glob

try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    storage = None


@contextlib.contextmanager
def open_sharded_file(path: str, mode: str = "r", encoding: Optional[str] = "utf-8") -> Iterator[io.IOBase]:
    """Open a file from GCS (gs://) or local filesystem."""

    if path.startswith("gs://"):
        if storage is None:
            raise RuntimeError("google-cloud-storage is required for GCS paths")
        bucket_name, blob_path = path[5:].split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        binary = "b" in mode
        if "r" in mode:
            if binary:
                data = blob.download_as_bytes()
                fh = io.BytesIO(data)
            else:
                data = blob.download_as_text(encoding=encoding)
                fh = io.StringIO(data)
            try:
                yield fh
            finally:
                fh.close()
        else:
            buffer: io.IOBase
            if binary:
                buffer = io.BytesIO()
            else:
                buffer = io.StringIO()
            try:
                yield buffer
                payload = buffer.getvalue()
                if binary:
                    blob.upload_from_string(payload)
                else:
                    blob.upload_from_string(payload)
            finally:
                buffer.close()
    else:
        path_obj = Path(path)
        if "w" in mode:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open(mode, encoding=encoding) as fh:
            yield fh


def write_jsonl(path: str, records: Iterable[dict]) -> None:
    with open_sharded_file(path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def path_exists(path: str) -> bool:
    if path.startswith("gs://"):
        if storage is None:
            return False
        bucket_name, blob_path = path[5:].split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.exists()
    return Path(path).exists()


def _gcs_list(pattern: str) -> list[str]:
    if storage is None:
        raise RuntimeError("google-cloud-storage is required for GCS paths")
    bucket_name, blob_pattern = pattern[5:].split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    wildcard_chars = ["*", "?", "["]
    prefix = blob_pattern
    for char in wildcard_chars:
        idx = blob_pattern.find(char)
        if idx != -1:
            prefix = blob_pattern[:idx]
            break
    blobs = bucket.list_blobs(prefix=prefix)
    matches = []
    for blob in blobs:
        if fnmatch.fnmatch(blob.name, blob_pattern):
            matches.append(f"gs://{bucket_name}/{blob.name}")
    if not matches and not any(ch in blob_pattern for ch in wildcard_chars):
        if bucket.blob(blob_pattern).exists():
            matches.append(pattern)
    return sorted(matches)


def resolve_glob_paths(pattern: str) -> list[str]:
    if pattern.startswith("gs://"):
        return _gcs_list(pattern)
    paths = glob.glob(pattern)
    if not paths and not any(ch in pattern for ch in "*?["):
        if Path(pattern).exists():
            paths = [pattern]
    return sorted(paths)


__all__ = ["open_sharded_file", "write_jsonl", "path_exists", "resolve_glob_paths"]
