import argparse
import glob
import os
import sys
import subprocess
import tempfile
import urllib.request
from typing import List, Tuple

FA_GCS_FLAG = "--fa_wheel_gcs_uri"
FA_URL_FLAG = "--fa_wheel_url"
USE_FA_FLAG = "--use_flash_attn"

def _pip_install(path: str) -> None:
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", path, "--no-deps"],
            check=False,
        )
        print(f"[launcher] Pip install attempted (non-fatal): {path}")
    except Exception as e:
        print(f"[launcher] Pip install failed non-fatally: {e}")

def _download_gcs(gcs_uri: str, dst_path: str) -> bool:
    try:
        from google.cloud import storage
        if not gcs_uri.startswith("gs://"):
            raise ValueError("GCS URI must start with gs://")
        _, rem = gcs_uri.split("gs://", 1)
        bucket_name, blob_name = rem.split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(dst_path)
        print(f"[launcher] Downloaded FA wheel from GCS to {dst_path}")
        return True
    except Exception as e:
        print(f"[launcher] GCS download failed (non-fatal): {e}")
        return False

def _download_http(url: str, dst_path: str) -> bool:
    try:
        with urllib.request.urlopen(url) as r, open(dst_path, "wb") as f:
            f.write(r.read())
        print(f"[launcher] Downloaded FA wheel from URL to {dst_path}")
        return True
    except Exception as e:
        print(f"[launcher] HTTP download failed (non-fatal): {e}")
        return False

def _maybe_install_fa(enable: bool, gcs_uri: str, http_url: str) -> None:
    if not enable:
        print("[launcher] FlashAttention disabled by flag.")
        return
    tmp = os.path.join(tempfile.gettempdir(), "flash_attn.whl")
    if gcs_uri and _download_gcs(gcs_uri, tmp):
        _pip_install(tmp); return
    if http_url and _download_http(http_url, tmp):
        _pip_install(tmp); return
    wheels = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "wheels", "flash_attn-*.whl")))
    if wheels:
        _pip_install(wheels[-1]); return
    print("[launcher] No FA wheel available; continuing without FlashAttention.")

def _parse_launcher_flags(argv: List[str]) -> Tuple[bool, str, str]:
    """
    Parse only the launcher's flags from argv (does not error on unknowns).
    Returns: (use_fa, fa_gcs_uri, fa_url)
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(USE_FA_FLAG, type=str, default=os.getenv("STAGE1_USE_FLASH_ATTN", "false"))
    parser.add_argument(FA_GCS_FLAG, type=str, default=os.getenv("FA_WHEEL_GCS_URI", ""))
    parser.add_argument(FA_URL_FLAG, type=str, default=os.getenv("FA_WHEEL_URL", ""))
    known, _ = parser.parse_known_args(argv)
    use_fa = str(known.use_flash_attn).lower() in {"1", "true", "yes", "y"}
    return use_fa, known.fa_wheel_gcs_uri, known.fa_wheel_url

def _filtered_argv_for_entrypoint(argv: List[str]) -> List[str]:
    """
    Remove the launcher's private flags (and their values) before passing to Stage_1.vertex.entrypoint.
    Supports both '--flag value' and '--flag=value'.
    Keeps --use_flash_attn, assuming Stage_1.vertex.entrypoint accepts it. If your entrypoint does NOT accept it,
    toggle KEEP_USE_FA_FLAG = False below to strip it and rely on env var instead.
    """
    KEEP_USE_FA_FLAG = True  # set False if Stage_1.vertex.entrypoint doesn't accept --use_flash_attn

    def is_flag_with_value(tok: str, name: str) -> bool:
        return tok == name or tok.startswith(name + "=")

    out: List[str] = []
    skip_next = False
    i = 0
    while i < len(argv):
        tok = argv[i]
        if skip_next:
            skip_next = False
            i += 1
            continue

        if is_flag_with_value(tok, FA_GCS_FLAG) or is_flag_with_value(tok, FA_URL_FLAG):
            if "=" not in tok and (i + 1) < len(argv) and not argv[i+1].startswith("--"):
                skip_next = True
            i += 1
            continue

        if not KEEP_USE_FA_FLAG and is_flag_with_value(tok, USE_FA_FLAG):
            if "=" not in tok and (i + 1) < len(argv) and not argv[i+1].startswith("--"):
                skip_next = True
            i += 1
            continue

        out.append(tok)
        i += 1

    return out

def main():
    orig_argv = sys.argv[1:]
    use_fa, fa_gcs_uri, fa_url = _parse_launcher_flags(orig_argv)

    os.environ.setdefault("STAGE1_USE_FLASH_ATTN", "1" if use_fa else "0")

    _maybe_install_fa(use_fa, fa_gcs_uri, fa_url)

    filtered = _filtered_argv_for_entrypoint(orig_argv)

    command = ["python3", "-m", "Stage_1.vertex.entrypoint", *filtered]
    os.execvp("python3", command)

if __name__ == "__main__":
    main()
