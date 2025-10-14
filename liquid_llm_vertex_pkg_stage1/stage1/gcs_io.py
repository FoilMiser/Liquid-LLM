"""Helpers for interacting with Google Cloud Storage via gcloud."""

import logging
import subprocess
from pathlib import Path
from typing import Iterable, List

LOGGER = logging.getLogger(__name__)


def gcs_cp(src: str, dst: str, recursive: bool = False) -> None:
    cmd = ["gcloud", "storage", "cp"]
    if recursive:
        cmd.append("-r")
    cmd.extend([src, dst])
    LOGGER.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_to_path(gcs_uri: str, local_path: Path) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    gcs_cp(gcs_uri, str(local_path))
    return local_path


def upload_file(local_path: Path, gcs_uri: str) -> None:
    gcs_cp(str(local_path), gcs_uri)


def upload_logs_periodically(log_path: Path, gcs_uri: str) -> None:
    if not log_path.exists():
        LOGGER.debug("Log path %s does not exist; skipping upload.", log_path)
        return
    upload_file(log_path, gcs_uri)


def list_gcs(glob_uri: str) -> List[str]:
    cmd = ["gcloud", "storage", "ls", glob_uri]
    LOGGER.info("Listing GCS files: %s", glob_uri)
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def download_many(gcs_uris: Iterable[str], dst_dir: Path) -> List[Path]:
    paths = []
    for uri in gcs_uris:
        local = dst_dir / Path(uri).name
        download_to_path(uri, local)
        paths.append(local)
    return paths


__all__ = [
    "gcs_cp",
    "download_to_path",
    "upload_file",
    "upload_logs_periodically",
    "list_gcs",
    "download_many",
]
