"""Vertex AI entrypoint that adapts Stage-1 configuration to Vertex arguments."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import uuid
from typing import List, Sequence

from Stage_1.cli import main as stage1_main
from Stage_1.utils import get_hf_token

_LOG_PREFIX = "[Stage_1.vertex.entrypoint]"

_TOKEN_ENV_VARS = ("HF_TOKEN", "HF_API_TOKEN", "HUGGINGFACEHUB_API_TOKEN")


def _parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Stage-1 Vertex AI adapter",
        add_help=True,
    )
    parser.add_argument("--resume_gcs_uri", type=str, default=None)
    parser.add_argument("--output_gcs_uri", type=str, default=None)
    parser.add_argument("--teacher_name", type=str, default=None)
    parser.add_argument("--teacher_endpoint", type=str, default=None)
    parser.add_argument("--teacher_max_batch_size", type=int, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_cfg", type=str, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--train_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--betas", type=str, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--throughput_tokens", type=int, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hf_secret_name", type=str, default=None)
    parser.add_argument("--hf_secret_project", type=str, default=None)
    parser.add_argument("--hf_token_value", type=str, default=None)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    known, extras = parser.parse_known_args(argv)
    return known, extras


def _append_flag(args: List[str], name: str, value: object | None) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        literal = "true" if value else "false"
    else:
        literal = str(value)
    args.append(f"--{name}={literal}")


def _teacher_alias(model_id: str | None) -> str | None:
    if not model_id:
        return None
    lowered = model_id.lower()
    if "meta-llama-3.1-8b-instruct" in lowered:
        return "llama-3.1-8b-instruct"
    if "meta-llama-3.1-8b" in lowered:
        return "llama-3.1-8b"
    return None


def _resolve_output_uri(cli_output: str | None) -> str | None:
    if cli_output:
        return cli_output
    for env_key in ("AIP_CHECKPOINT_DIR", "AIP_MODEL_DIR", "AIP_OUTPUT_DIR"):
        value = os.getenv(env_key)
        if value:
            return value
    return None


def _resolve_project_hint(cli_value: str | None) -> str | None:
    if cli_value:
        return cli_value
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


def _ensure_hf_token(secret_name: str | None, explicit: str | None, project_hint: str | None) -> None:
    token = explicit
    if token is None and secret_name:
        try:
            token = get_hf_token(secret_name, project_id=project_hint)
        except ValueError as exc:
            print(
                f"{_LOG_PREFIX} WARNING: unable to resolve secret '{secret_name}': {exc}",
                file=sys.stderr,
            )
        except Exception as exc:  # pragma: no cover - networking errors
            print(
                f"{_LOG_PREFIX} WARNING: failed to fetch secret '{secret_name}': {exc}",
                file=sys.stderr,
            )
    if not token:
        return
    for env_key in _TOKEN_ENV_VARS:
        os.environ.setdefault(env_key, token)


def _maybe_warn_unused(label: str, value: object | None) -> None:
    if value is None:
        return
    print(f"{_LOG_PREFIX} NOTE: ignoring argument --{label}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    known, extras = _parse_args(argv)
    stage1_args: list[str] = []

    _append_flag(stage1_args, "resume_gcs_uri", known.resume_gcs_uri)
    output_uri = _resolve_output_uri(known.output_gcs_uri)
    _append_flag(stage1_args, "output_gcs_uri", output_uri)
    _append_flag(stage1_args, "batch_size", known.batch_size)
    _append_flag(stage1_args, "eval_every", known.eval_every)
    _append_flag(stage1_args, "save_every", known.save_every)
    _append_flag(stage1_args, "lr", known.lr)
    _append_flag(stage1_args, "weight_decay", known.weight_decay)
    _append_flag(stage1_args, "betas", known.betas)
    _append_flag(stage1_args, "warmup_steps", known.warmup_steps)
    _append_flag(stage1_args, "precision", known.dtype)
    _append_flag(stage1_args, "device", known.device)
    _append_flag(stage1_args, "hf_secret_name", known.hf_secret_name)
    _append_flag(stage1_args, "hf_cache_dir", known.hf_cache_dir)
    _append_flag(stage1_args, "seed", known.seed)
    _append_flag(stage1_args, "log_every", known.log_every)
    _append_flag(stage1_args, "num_workers", known.num_workers)
    _append_flag(stage1_args, "gradient_accumulation_steps", known.gradient_accumulation_steps)
    _append_flag(stage1_args, "max_grad_norm", known.max_grad_norm)

    if known.block_size is not None:
        _append_flag(stage1_args, "seq_len", known.block_size)
    if known.train_steps is not None:
        _append_flag(stage1_args, "max_steps", known.train_steps)
    if known.dataset_cfg is not None:
        _append_flag(stage1_args, "dataset_cfg", known.dataset_cfg)

    if known.teacher_name:
        alias = _teacher_alias(known.teacher_name)
        if alias:
            _append_flag(stage1_args, "teacher", alias)
        _append_flag(stage1_args, "teacher_id", known.teacher_name)
    _append_flag(stage1_args, "teacher_endpoint", known.teacher_endpoint)
    _append_flag(stage1_args, "teacher_max_batch_size", known.teacher_max_batch_size)

    run_id = known.run_id or os.getenv("AIP_TRAINING_JOB_ID") or os.getenv("AIP_JOB_ID")
    if not run_id:
        run_id = os.getenv("AIP_TRIAL_ID") or os.getenv("CLOUD_ML_JOB_ID")
    if not run_id:
        run_id = str(uuid.uuid4())
    _append_flag(stage1_args, "run_id", run_id)

    if known.throughput_tokens is not None:
        _maybe_warn_unused("throughput_tokens", known.throughput_tokens)
    if known.dataset_name:
        _maybe_warn_unused("dataset_name", known.dataset_name)
    if known.dataset_config:
        _maybe_warn_unused("dataset_config", known.dataset_config)

    project_hint = _resolve_project_hint(known.hf_secret_project)
    _ensure_hf_token(known.hf_secret_name, known.hf_token_value, project_hint)

    stage1_args.extend(extras)

    cmd = " ".join(shlex.quote(arg) for arg in stage1_args)
    print(f"{_LOG_PREFIX} Launching Stage-1 CLI with arguments: {cmd}")
    stage1_main(stage1_args)


if __name__ == "__main__":  # pragma: no cover
    main()
