"""Helpers for interacting with Google Cloud Storage in Vertex jobs."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, List

try:  # pragma: no cover - optional dependency
    import gcsfs
except Exception:  # pragma: no cover - handled at runtime
    gcsfs = None

from .utils import Backoff, configure_logging

logger = configure_logging()


class GCSIOError(RuntimeError):
    """Raised when a GCS operation fails after retries."""


_DEF_RETRIES = 3


def _has_gcloud() -> bool:
    return shutil.which("gcloud") is not None


def _run_command(cmd: List[str]) -> subprocess.CompletedProcess:
    logger.debug("Executing command: %s", " ".join(cmd))
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _with_retries(func, *, retries: int = _DEF_RETRIES) -> Any:
    backoff = Backoff()
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as exc:  # pragma: no cover - runtime only
            last_error = exc
            logger.warning("GCS operation failed (attempt %s/%s): %s", attempt + 1, retries + 1, exc)
            if attempt >= retries:
                break
            backoff.sleep()
    if last_error is None:
        raise GCSIOError("GCS operation failed")
    raise GCSIOError(str(last_error))


def _gcsfs_filesystem():
    if gcsfs is None:
        raise GCSIOError("gcsfs is required but not installed; install gcsfs or ensure gcloud is available")
    return gcsfs.GCSFileSystem()


def gcs_to_local(gcs_uri: str, local_path: str, retries: int = _DEF_RETRIES) -> str:
    """Copy a file from GCS to the local path."""

    _ensure_parent(local_path)

    if _has_gcloud():
        def _copy_cmd() -> str:
            result = _run_command(["gcloud", "storage", "cp", gcs_uri, local_path])
            if result.returncode != 0:
                raise GCSIOError(result.stderr)
            logger.info("Copied %s -> %s", gcs_uri, local_path)
            return local_path

        return _with_retries(_copy_cmd, retries=retries)

    def _copy_fs() -> str:
        fs = _gcsfs_filesystem()
        with fs.open(gcs_uri, "rb") as src, open(local_path, "wb") as dst:
            dst.write(src.read())
        logger.info("Copied (gcsfs) %s -> %s", gcs_uri, local_path)
        return local_path

    return _with_retries(_copy_fs, retries=retries)


def local_to_gcs(local_path: str, gcs_uri: str, retries: int = _DEF_RETRIES) -> None:
    """Upload a local file or directory to GCS with verification."""

    src_path = Path(local_path)

    def _upload_file(src: Path, dest: str) -> None:
        if _has_gcloud():
            def _cmd() -> None:
                result = _run_command(["gcloud", "storage", "cp", str(src), dest])
                if result.returncode != 0:
                    raise GCSIOError(result.stderr)
                verify = _run_command(["gcloud", "storage", "ls", dest])
                if verify.returncode != 0:
                    raise GCSIOError(verify.stderr)
                logger.info("Uploaded %s -> %s", src, dest)

            _with_retries(_cmd, retries=retries)
            return

        def _upload_fs() -> None:
            fs = _gcsfs_filesystem()
            with open(src, "rb") as src_handle, fs.open(dest, "wb") as dst_handle:
                dst_handle.write(src_handle.read())
            try:
                fs.info(dest)
            except Exception as exc:  # pragma: no cover - runtime verification
                raise GCSIOError(f"Failed to verify upload for {dest}: {exc}")
            logger.info("Uploaded (gcsfs) %s -> %s", src, dest)

        _with_retries(_upload_fs, retries=retries)

    if src_path.is_dir():
        for child in sorted(src_path.rglob("*")):
            if child.is_file():
                rel = child.relative_to(src_path)
                dest = f"{gcs_uri.rstrip('/')}/{rel.as_posix()}"
                _upload_file(child, dest)
    else:
        _upload_file(src_path, gcs_uri)


def list_gcs(uri: str, retries: int = _DEF_RETRIES) -> List[str]:
    """List objects that match the provided GCS URI."""

    if _has_gcloud():
        def _list_cmd() -> List[str]:
            result = _run_command(["gcloud", "storage", "ls", uri])
            if result.returncode != 0:
                raise GCSIOError(result.stderr)
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]

        return _with_retries(_list_cmd, retries=retries)

    def _list_fs() -> List[str]:
        fs = _gcsfs_filesystem()
        matches = fs.glob(uri)
        return _normalize_gcsfs_listing(matches)

    return _with_retries(_list_fs, retries=retries)


def _normalize_gcsfs_listing(matches: Iterable[str]) -> List[str]:
    results: List[str] = []
    for match in matches:
        if match.startswith("gs://"):
            results.append(match)
        else:
            results.append(f"gs://{match}")
    return results


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
