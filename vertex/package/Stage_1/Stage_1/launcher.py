"""Vertex AI launcher that optionally installs FlashAttention at runtime."""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile
import urllib.request


def _pip_install(path: str) -> None:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", path, "--no-deps"], check=False)
        print(f"[launcher] Pip install attempted (non-fatal): {path}")
    except Exception as exc:  # pragma: no cover - defensive logging only
        print(f"[launcher] Pip install failed non-fatally: {exc}")


def _download_gcs(gcs_uri: str, dst_path: str) -> bool:
    try:
        from google.cloud import storage

        if not gcs_uri.startswith("gs://"):
            raise ValueError("GCS URI must start with gs://")
        _, remainder = gcs_uri.split("gs://", 1)
        bucket_name, blob_name = remainder.split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(dst_path)
        print(f"[launcher] Downloaded FA wheel from GCS to {dst_path}")
        return True
    except Exception as exc:  # pragma: no cover - network dependency
        print(f"[launcher] GCS download failed (non-fatal): {exc}")
        return False


def _download_http(url: str, dst_path: str) -> bool:
    try:
        with urllib.request.urlopen(url) as response, open(dst_path, "wb") as handle:
            handle.write(response.read())
        print(f"[launcher] Downloaded FA wheel from URL to {dst_path}")
        return True
    except Exception as exc:  # pragma: no cover - network dependency
        print(f"[launcher] HTTP download failed (non-fatal): {exc}")
        return False


def maybe_install_flash_attn(enable: bool, gcs_uri: str | None, url: str | None) -> None:
    if not enable:
        print("[launcher] FlashAttention disabled by flag.")
        return

    wheel_path = os.path.join(tempfile.gettempdir(), "flash_attn.whl")
    if gcs_uri and _download_gcs(gcs_uri, wheel_path):
        _pip_install(wheel_path)
        return

    if url and _download_http(url, wheel_path):
        _pip_install(wheel_path)
        return

    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    wheels = sorted(glob.glob(os.path.join(package_root, "wheels", "flash_attn-*.whl")))
    if wheels:
        _pip_install(wheels[-1])
        return

    print("[launcher] No FA wheel available; continuing without FlashAttention.")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--use_flash_attn", type=str, default="false")
    parser.add_argument("--fa_wheel_gcs_uri", type=str, default="")
    parser.add_argument("--fa_wheel_url", type=str, default="")
    known, _ = parser.parse_known_args()

    enable = str(known.use_flash_attn).lower() in {"1", "true", "yes", "y"}
    gcs_uri = known.fa_wheel_gcs_uri or None
    url = known.fa_wheel_url or None
    maybe_install_flash_attn(enable, gcs_uri, url)

    os.execv(sys.executable, [sys.executable, "-m", "Stage_1.cli", *sys.argv[1:]])


if __name__ == "__main__":  # pragma: no cover - CLI module
    main()
