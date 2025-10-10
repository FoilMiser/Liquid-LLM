"""GCS/local IO helpers."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

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


__all__ = ["open_sharded_file", "write_jsonl"]
