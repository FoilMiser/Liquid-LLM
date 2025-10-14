"""Runtime setup utilities for Vertex AI jobs."""

import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import login

LOGGER = logging.getLogger(__name__)


def login_to_huggingface() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable must be set for Hugging Face login.")
    login(token=token, add_to_git_credential=False)
    LOGGER.info("Authenticated to Hugging Face Hub.")


def download_flashattention(wheel_gcs_uri: str, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = dst_dir / Path(wheel_gcs_uri).name
    if wheel_path.exists():
        LOGGER.info("FlashAttention wheel already present at %s", wheel_path)
        return wheel_path
    cmd = [
        "gcloud",
        "storage",
        "cp",
        wheel_gcs_uri,
        str(wheel_path),
    ]
    LOGGER.info("Downloading FlashAttention wheel from %s", wheel_gcs_uri)
    subprocess.run(cmd, check=True)
    return wheel_path


def install_wheel(wheel_path: Path) -> None:
    LOGGER.info("Installing wheel %s", wheel_path)
    subprocess.run([sys.executable, "-m", "pip", "install", str(wheel_path)], check=True)


def try_import_flashattention() -> Optional[object]:
    try:
        return importlib.import_module("flash_attn")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("FlashAttention import failed: %s", exc)
        return None


def configure_attention_backend(flash_mod: Optional[object]) -> None:
    if not torch.cuda.is_available():
        LOGGER.warning("CUDA not available; attention backend configuration skipped.")
        return
    if flash_mod is not None:
        LOGGER.info("FlashAttention available: %s", flash_mod.__name__)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        return

    LOGGER.info("Falling back to PyTorch SDPA backends.")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)


def setup_runtime(wheel_gcs_uri: str, precision: str = "bfloat16") -> torch.dtype:
    login_to_huggingface()
    wheel_dir = Path("/tmp/wheels")
    try:
        wheel_path = download_flashattention(wheel_gcs_uri, wheel_dir)
        install_wheel(wheel_path)
    except subprocess.CalledProcessError as exc:
        LOGGER.warning("Failed to install FlashAttention: %s", exc)
    flash_mod = try_import_flashattention()
    configure_attention_backend(flash_mod)
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    LOGGER.info("Using precision %s (dtype=%s)", precision, dtype)
    return dtype


def autocast_context(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    return torch.cuda.amp.autocast(dtype=torch.float16)


__all__ = [
    "setup_runtime",
    "autocast_context",
    "login_to_huggingface",
    "download_flashattention",
    "install_wheel",
    "try_import_flashattention",
    "configure_attention_backend",
]
