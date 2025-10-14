"""Helpers for interacting with Google Cloud Storage in Vertex jobs."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

from .utils import Backoff, configure_logging

logger = configure_logging()


class GCSIOError(RuntimeError):
    """Raised when a GCS operation fails after retries."""


_DEF_RETRIES = 3


def _run_command(cmd: List[str]) -> subprocess.CompletedProcess:
    logger.debug("Executing command: %s", " ".join(cmd))
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def gcs_to_local(gcs_uri: str, local_path: str, retries: int = _DEF_RETRIES) -> str:
    """Copy a file from GCS to the local path."""

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    backoff = Backoff()
    for attempt in range(retries + 1):
        result = _run_command(["gcloud", "storage", "cp", gcs_uri, local_path])
        if result.returncode == 0:
            logger.info("Copied %s -> %s", gcs_uri, local_path)
            return local_path
        logger.warning("Failed to copy %s (attempt %s/%s): %s", gcs_uri, attempt + 1, retries + 1, result.stderr)
        if attempt >= retries:
            raise GCSIOError(f"Failed to copy {gcs_uri} after {retries + 1} attempts: {result.stderr}")
        backoff.sleep()
    raise AssertionError("Unreachable")


def local_to_gcs(local_path: str, gcs_uri: str, retries: int = _DEF_RETRIES) -> None:
    """Upload a local file or directory to GCS."""

    backoff = Backoff()
    for attempt in range(retries + 1):
        result = _run_command(["gcloud", "storage", "cp", "-r", local_path, gcs_uri])
        if result.returncode == 0:
            logger.info("Uploaded %s -> %s", local_path, gcs_uri)
            return
        logger.warning("Failed to upload %s (attempt %s/%s): %s", local_path, attempt + 1, retries + 1, result.stderr)
        if attempt >= retries:
            raise GCSIOError(f"Failed to upload {local_path}: {result.stderr}")
        backoff.sleep()


def list_gcs(uri: str, retries: int = _DEF_RETRIES) -> List[str]:
    """List objects that match the provided GCS URI."""

    backoff = Backoff()
    for attempt in range(retries + 1):
        result = _run_command(["gcloud", "storage", "ls", uri])
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
        logger.warning("Failed to list %s (attempt %s/%s): %s", uri, attempt + 1, retries + 1, result.stderr)
        if attempt >= retries:
            raise GCSIOError(f"Failed to list {uri}: {result.stderr}")
        backoff.sleep()
    raise AssertionError("Unreachable")


def maybe_sync_dir(local_dir: str, gcs_dir: str) -> None:
    """Optionally upload files if a GCS destination is provided."""

    if not gcs_dir:
        logger.debug("Skipping sync for %s; no destination provided", local_dir)
        return
    tmp_dir = Path(local_dir)
    if not tmp_dir.exists():
        logger.warning("Local directory %s missing; nothing to sync", local_dir)
        return
    local_to_gcs(str(local_dir), gcs_dir)


def ensure_local_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
