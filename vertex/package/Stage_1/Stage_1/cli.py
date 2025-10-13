"""CLI for Stage-1 training."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib import resources

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from .trainer import Stage1Trainer
from .utils import (
    DEFAULT_DATASET_CFG,
    build_arg_parser,
    build_config,
    dump_config,
    ensure_output_path,
    get_hf_token,
    path_exists,
)
from .utils.attention import _have_flash_attn

CANONICAL_TEACHER = "meta-llama/Meta-Llama-3.1-8B"
TOKEN_ENV_KEYS = ("HF_TOKEN", "HF_API_TOKEN", "HUGGINGFACEHUB_API_TOKEN")


def _find_flash_wheel() -> str | None:
    try:
        wheels_dir = resources.files("Stage_1").joinpath("wheels")
    except Exception:
        return None
    try:
        for entry in wheels_dir.iterdir():
            if entry.name.endswith(".whl"):
                return str(entry)
    except FileNotFoundError:
        return None
    return None


def _ensure_flash_attention(enable_flash: bool, allow_install: bool) -> bool:
    if not enable_flash:
        return False
    if _have_flash_attn():
        return True
    if not allow_install:
        return False
    wheel_path = _find_flash_wheel()
    if not wheel_path:
        print("FlashAttention wheel not packaged; falling back to SDPA.")
        return False
    print(f"Attempting local FlashAttention install from {wheel_path}...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                wheel_path,
                "--no-deps",
                "--no-warn-script-location",
            ],
            check=False,
        )
    except Exception as exc:
        print(f"FlashAttention wheel install failed: {exc}. Falling back to SDPA.")
        return False
    if _have_flash_attn():
        return True
    print("FlashAttention remains unavailable after install attempt; using SDPA.")
    return False


def _validate_teacher(name: str) -> None:
    if name != CANONICAL_TEACHER:
        raise SystemExit(
            "Stage-1 distillation requires --teacher_name=meta-llama/Meta-Llama-3.1-8B."
        )

    token = None
    for env_key in TOKEN_ENV_KEYS:
        token = os.getenv(env_key)
        if token:
            break

    api = HfApi(token=token)
    try:
        api.model_info(name)
    except RepositoryNotFoundError as exc:  # pragma: no cover - network call
        raise SystemExit(
            "Teacher model 'meta-llama/Meta-Llama-3.1-8B' is not available on Hugging Face Hub."
        ) from exc
    except HfHubHTTPError as exc:  # pragma: no cover - network call
        status = getattr(exc.response, "status_code", None)
        if status in {401, 403}:
            raise SystemExit(
                "Access to 'meta-llama/Meta-Llama-3.1-8B' was denied. Ensure your HF token has the required permissions."
            ) from exc
        raise SystemExit(
            f"Failed to validate teacher model 'meta-llama/Meta-Llama-3.1-8B': {exc}"
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(
            "Unable to reach Hugging Face Hub to validate the teacher model."
        ) from exc


def _detect_project_id() -> str | None:
    for env_key in (
        "AIP_PROJECT_ID",
        "GOOGLE_CLOUD_PROJECT",
        "PROJECT_ID",
        "CLOUD_ML_PROJECT_ID",
        "GCLOUD_PROJECT",
    ):
        value = os.getenv(env_key)
        if value:
            return value
    return None


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = build_config(args)
    if not config.resume_gcs_uri:
        raise SystemExit(
            "Stage-1 requires --resume_gcs_uri pointing to the post-surgery checkpoint (e.g. gs://liquid-llm-bucket-2/stage1/stage1.pt)."
        )
    if not path_exists(config.resume_gcs_uri):
        raise SystemExit(f"Checkpoint not found at {config.resume_gcs_uri}")
    if not config.dataset_cfg:
        raise SystemExit(
            f"Stage-1 requires --dataset_cfg pointing to a manifest JSONL (default: {DEFAULT_DATASET_CFG})."
        )
    if not path_exists(config.dataset_cfg):
        raise SystemExit(f"Dataset manifest not found at {config.dataset_cfg}")

    token = None
    if not any(os.getenv(env_key) for env_key in TOKEN_ENV_KEYS):
        secret_name = getattr(config, "hf_secret_name", None)
        if secret_name:
            try:
                token = get_hf_token(secret_name, project_id=_detect_project_id())
            except ValueError as exc:
                raise SystemExit(f"Unable to resolve HF secret '{secret_name}': {exc}") from exc
            except Exception as exc:  # pragma: no cover - secret manager/network errors
                raise SystemExit(
                    f"Failed to fetch Hugging Face token from secret '{secret_name}': {exc}"
                ) from exc
    if token:
        for env_key in TOKEN_ENV_KEYS:
            os.environ.setdefault(env_key, token)

    flash_enabled = bool(config.use_flash_attn)
    allow_install = bool(getattr(config, "flash_wheel_install", True))
    have_flash = _ensure_flash_attention(flash_enabled, allow_install)
    if flash_enabled and not have_flash:
        print("Warning: FlashAttention unavailable; falling back to SDPA.")
    config.use_flash_attn = bool(flash_enabled and have_flash)

    _validate_teacher(config.teacher_name)
    if not config.run_id:
        config.run_id = datetime.now(timezone.utc).strftime("stage1-%Y%m%d-%H%M%S")
    if config.output_gcs_uri:
        base_uri = config.output_gcs_uri.rstrip("/")
        config.output_gcs_uri = f"{base_uri}/{config.run_id}"
    if config.output_gcs_uri and not config.output_gcs_uri.startswith("gs://"):
        ensure_output_path(config.output_gcs_uri)
    backend = "flash" if config.use_flash_attn and _have_flash_attn() else "sdpa"
    derived_grad = getattr(config, "extra", {}).get("derived_grad_accum_steps")
    if derived_grad is not None:
        print(
            "Adjusted gradient_accumulation_steps to "
            f"{config.gradient_accumulation_steps} for throughput target"
        )
    print(
        "resolved_config: "
        f"seq_len={config.seq_len}, "
        f"block_size={config.block_size}, "
        f"batch_size={config.batch_size}, "
        f"grad_accum_steps={config.gradient_accumulation_steps}, "
        f"throughput_tokens={config.throughput_tokens}"
    )
    print(f"attention_backend={backend}")
    trainer = Stage1Trainer(config)
    config_path = os.path.join(trainer.output_path, "config_stage1.json")
    dump_config(config_path, config)
    trainer.train()


if __name__ == "__main__":  # pragma: no cover
    main()
