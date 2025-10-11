"""Vertex AI entrypoint for Stage-1 training."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Sequence

import pkg_resources

from Stage_1.logging_utils import get_logger
from Stage_1.utils import (
    DEFAULT_DATASET_CFG,
    build_arg_parser,
    build_config,
    dump_config,
    ensure_output_path,
    get_hf_token,
    path_exists,
)

_LOGGER = get_logger("vertex.entrypoint")
_CANONICAL_TEACHER = "meta-llama/Meta-Llama-3.1-8B"
_TOKEN_ENV_VARS = (
    "HF_TOKEN",
    "HF_API_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
)


def _log_dependency_versions() -> None:
    for name in ["datasets", "fsspec", "gcsfs"]:
        try:
            importlib.import_module(name)
            version = pkg_resources.get_distribution(name).version
            logging.info("Resolved %s==%s", name, version)
        except Exception as exc:  # pragma: no cover - best effort logging
            logging.warning("Could not resolve %s version: %s", name, exc)


def _validate_dependency_stack() -> None:
    try:
        datasets_version = pkg_resources.get_distribution("datasets").version
    except Exception:
        return

    if datasets_version != "2.20.0":
        return

    try:
        fsspec_version = pkg_resources.get_distribution("fsspec").version
    except Exception as exc:
        raise EntrypointError(
            "datasets==2.20.0 requires fsspec<=2024.5.0; ensure the worker image provides fsspec==2024.5.0."
        ) from exc

    if pkg_resources.parse_version(fsspec_version) > pkg_resources.parse_version("2024.5.0"):
        raise EntrypointError(
            "Incompatible dependency stack detected: datasets==2.20.0 expects fsspec<=2024.5.0, but fsspec=="
            f"{fsspec_version} was resolved. Update the environment to use fsspec==2024.5.0 or adjust the dataset version before training."
        )


class EntrypointError(Exception):
    """Base error for entrypoint failures."""


class UserInputError(EntrypointError):
    """Raised for recoverable user input issues."""


class FlashAttentionError(EntrypointError):
    """Raised when FlashAttention setup fails."""


def _bool_flag(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _build_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    if "fa_wheel_gcs_uri" not in {action.dest for action in parser._actions}:  # type: ignore[attr-defined]
        parser.add_argument("--fa_wheel_gcs_uri", type=str, default=None, help="GCS URI to a FlashAttention wheel")
    if "allow_fa_fallback" not in {action.dest for action in parser._actions}:  # type: ignore[attr-defined]
        parser.add_argument(
            "--allow_fa_fallback",
            type=_bool_flag,
            default=False,
            help="Continue without FlashAttention if installation fails",
        )
    return parser


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


def _export_hf_token(token: str) -> None:
    token = token.strip()
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    for env_key in _TOKEN_ENV_VARS:
        os.environ.setdefault(env_key, token)


def _prepare_hf_token(secret_name: str | None, project_id: str | None) -> None:
    if not secret_name:
        token = get_hf_token(None, project_id=project_id)
        if token:
            _export_hf_token(token)
        return

    try:
        token = get_hf_token(secret_name, project_id=project_id)
    except ValueError as exc:
        raise UserInputError(f"Unable to resolve HF secret '{secret_name}': {exc}") from exc
    if not token:
        raise UserInputError(
            f"Hugging Face secret '{secret_name}' was not found; ensure it exists and the training service account has access."
        )
    _export_hf_token(token)


def _validate_teacher(name: str) -> None:
    if name != _CANONICAL_TEACHER:
        raise UserInputError(
            "Stage-1 distillation requires --teacher_name=meta-llama/Meta-Llama-3.1-8B."
        )

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    token = (
        os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HF_API_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    api = HfApi(token=token)
    try:
        api.model_info(name)
    except RepositoryNotFoundError as exc:  # pragma: no cover - external dependency
        raise UserInputError(
            "Teacher model 'meta-llama/Meta-Llama-3.1-8B' is not available on Hugging Face Hub."
        ) from exc
    except HfHubHTTPError as exc:  # pragma: no cover - external dependency
        status = getattr(exc.response, "status_code", None)
        if status in {401, 403}:
            raise UserInputError(
                "Access to 'meta-llama/Meta-Llama-3.1-8B' was denied. Ensure your HF token has the required permissions."
            ) from exc
        raise UserInputError(
            f"Failed to validate teacher model 'meta-llama/Meta-Llama-3.1-8B': {exc}"
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise UserInputError(
            "Unable to reach Hugging Face Hub to validate the teacher model."
        ) from exc


def _download_flash_attention_wheel(gcs_uri: str) -> str:
    if not gcs_uri.startswith("gs://"):
        raise FlashAttentionError("FlashAttention wheel must be provided via a gs:// URI.")
    try:
        _, remainder = gcs_uri.split("gs://", 1)
        bucket_name, blob_name = remainder.split("/", 1)
    except ValueError as exc:
        raise FlashAttentionError(f"Invalid FlashAttention URI '{gcs_uri}'.") from exc

    try:
        from google.cloud import storage
    except Exception as exc:  # pragma: no cover - dependency issues
        raise FlashAttentionError(f"google-cloud-storage is required to download {gcs_uri}: {exc}") from exc

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise FlashAttentionError(f"FlashAttention wheel not found at {gcs_uri}.")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(blob_name)[-1] or ".whl")
    os.close(tmp_fd)
    blob.download_to_filename(tmp_path)
    return tmp_path


def _install_flash_attention(wheel_uri: str) -> bool:
    try:
        wheel_path = _download_flash_attention_wheel(wheel_uri)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", wheel_path], check=True)
        _LOGGER.info("flash_attention_install", extra={"status": "success", "wheel": wheel_uri})
        return True
    except FlashAttentionError:
        raise
    except subprocess.CalledProcessError as exc:
        raise FlashAttentionError(f"Pip failed to install FlashAttention wheel from {wheel_uri}: {exc}") from exc
    except Exception as exc:  # pragma: no cover - network/filesystem failures
        raise FlashAttentionError(f"Unexpected error installing FlashAttention wheel from {wheel_uri}: {exc}") from exc
    finally:
        try:
            if "wheel_path" in locals() and os.path.exists(wheel_path):
                os.remove(wheel_path)
        except Exception:  # pragma: no cover - cleanup best-effort
            pass

def _prepare_flash_attention(use_flash_attn: bool, wheel_uri: str | None, allow_fallback: bool) -> bool:
    if not use_flash_attn or not wheel_uri:
        return use_flash_attn

    try:
        _install_flash_attention(wheel_uri)
        return True
    except FlashAttentionError as exc:
        if allow_fallback:
            _LOGGER.warning(
                "flash_attention_fallback",
                extra={"wheel": wheel_uri, "detail": str(exc)},
            )
            return False
        raise


def _ensure_paths(config) -> None:
    if not config.resume_gcs_uri:
        raise UserInputError(
            "Stage-1 requires --resume_gcs_uri pointing to the post-surgery checkpoint (e.g. gs://liquid-llm-bucket-2/stage1/stage1.pt)."
        )
    if config.resume_gcs_uri.startswith("gs://") and "/" not in config.resume_gcs_uri[5:]:
        raise UserInputError(
            f"Checkpoint URI must include both bucket and object path: {config.resume_gcs_uri}"
        )
    if not path_exists(config.resume_gcs_uri):
        raise UserInputError(f"Checkpoint not found at {config.resume_gcs_uri}")
    if not config.dataset_cfg:
        raise UserInputError(
            f"Stage-1 requires --dataset_cfg pointing to a manifest JSONL (default: {DEFAULT_DATASET_CFG})."
        )
    if config.dataset_cfg.startswith("gs://") and "/" not in config.dataset_cfg[5:]:
        raise UserInputError(
            f"Dataset manifest URI must include both bucket and object path: {config.dataset_cfg}"
        )
    if not path_exists(config.dataset_cfg):
        raise UserInputError(f"Dataset manifest not found at {config.dataset_cfg}")


def _finalize_output_path(config) -> None:
    if not config.run_id:
        config.run_id = datetime.now(timezone.utc).strftime("stage1-%Y%m%d-%H%M%S")
    if config.output_gcs_uri:
        base_uri = config.output_gcs_uri.rstrip("/")
        config.output_gcs_uri = f"{base_uri}/{config.run_id}"
        if not config.output_gcs_uri.startswith("gs://"):
            ensure_output_path(config.output_gcs_uri)


def _run_training(config) -> None:
    from Stage_1.trainer import Stage1Trainer

    trainer = Stage1Trainer(config)
    config_path = os.path.join(trainer.output_path, "config_stage1.json")
    dump_config(config_path, config)
    trainer.train()


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _log_dependency_versions()
    _validate_dependency_stack()

    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    try:
        config = build_config(args)
        _ensure_paths(config)

        project_id = _detect_project_id()
        _prepare_hf_token(getattr(args, "hf_secret_name", None), project_id)

        _validate_teacher(config.teacher_name)

        use_flash_attn = _prepare_flash_attention(
            bool(config.use_flash_attn), getattr(args, "fa_wheel_gcs_uri", None), bool(getattr(args, "allow_fa_fallback", False))
        )
        if use_flash_attn != bool(config.use_flash_attn):
            config.use_flash_attn = use_flash_attn

        _finalize_output_path(config)

        _LOGGER.info("stage1_launch", extra={"run_id": config.run_id, "output": config.output_gcs_uri})
        _run_training(config)
        return 0
    except FlashAttentionError as exc:
        _LOGGER.error("flash_attention_install_failed: %s", exc, extra={"error": str(exc)})
        return 1
    except UserInputError as exc:
        _LOGGER.error("argument_error: %s", exc, extra={"detail": str(exc)})
        return 1
    except KeyboardInterrupt:
        _LOGGER.warning("stage1_interrupted", extra={"signal": "keyboard"})
        return 130
    except EntrypointError as exc:
        _LOGGER.error("stage1_entrypoint_error: %s", exc, extra={"detail": str(exc)})
        return 1
    except Exception as exc:  # pragma: no cover - defensive catch-all
        _LOGGER.exception("stage1_unhandled_exception", extra={"error": str(exc)})
        return 1


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
