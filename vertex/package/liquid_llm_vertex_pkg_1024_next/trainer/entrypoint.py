"""Vertex AI entrypoint that adapts pipeline arguments to the toy trainer."""
from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence

from liquid_llm import train as train_module


def _coalesce(*values: Optional[str], default: Optional[str] = None) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip() == "":
                continue
            return value
    return default


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Liquid LLM Vertex launcher",
        allow_abbrev=False,
    )

    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dataset_name", default="wikitext")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=8)

    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--alpha_schedule", default=None)
    parser.add_argument("--temp_schedule", default=None)

    parser.add_argument("--lr_base", type=float, default=5e-5)
    parser.add_argument("--lr_peak", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr_scheduler", default="linear")

    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--fallback_save_every", type=int, default=200)
    parser.add_argument("--save_every_steps", type=int, default=100)
    parser.add_argument("--save_every_seconds", type=int, default=None)

    parser.add_argument("--gcs_root", default="")
    parser.add_argument("--output_gcs_uri", default=None)
    parser.add_argument("--resume_gcs_uri", default=None)

    parser.add_argument("--run_id", default=None)
    parser.add_argument("--phase_base", default="")
    parser.add_argument("--phase_index", type=int, default=0)

    parser.add_argument("--teacher_name", default="gpt2-xl")
    parser.add_argument("--teacher", dest="teacher_name")
    parser.add_argument("--teacher_precision", default="fp16")

    parser.add_argument("--train_steps", type=int, default=200)
    parser.add_argument("--improve_thresh", type=float, default=0.01)

    parser.add_argument("--precision", default=None)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--local_workdir", default=None)

    parser.add_argument("--eval_ctx_lens", default=None)
    parser.add_argument("--kd_scheme", default=None)
    parser.add_argument("--kl_scale", default=None)
    parser.add_argument("--save_best_on", default=None)
    parser.add_argument("--always_save_latest", default=None)
    parser.add_argument("--log_train_ppl", default=None)
    parser.add_argument("--selftest", default=None)
    parser.add_argument("--schedule_from_zero", action="store_true")
    parser.add_argument("--reset_lrsched_on_resume", action="store_true")
    parser.add_argument("--pipeline_mode", type=int, default=0)

    parser.add_argument("--hf_secret_name", default=None)
    parser.add_argument("--hf_token_value", default=None)
    parser.add_argument("--hf_token_file", default=None)
    parser.add_argument("--hf_token_gcs_uri", default=None)
    parser.add_argument("--require_hf_token", action="store_true")

    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--global_batch", type=int, default=None)
    parser.add_argument("--micro_batch", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--betas", type=float, nargs=2, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    return parser


def _dataset_name(args: argparse.Namespace) -> str:
    dataset = _coalesce(args.dataset, args.dataset_name, default="wikitext")
    if args.dataset_config:
        dataset = f"{dataset}/{args.dataset_config}"
    return dataset


def _batch_size(args: argparse.Namespace) -> int:
    value = args.batch if args.batch is not None else args.grad_accum
    return max(int(value), 1)


def _derive_run_id(args: argparse.Namespace) -> str:
    explicit = _coalesce(args.run_id)
    if explicit:
        return explicit
    if args.output_gcs_uri:
        base = os.path.basename(str(args.output_gcs_uri).rstrip("/"))
        if base:
            return base
    if args.phase_base:
        return f"{args.phase_base}-{args.phase_index}"
    return "vertex-run"


def _derive_output_dir(args: argparse.Namespace) -> str:
    return _coalesce(
        args.output_dir,
        args.local_workdir,
        os.environ.get("AIP_MODEL_DIR"),
        default="/tmp/liquid_llm",
    )


def _parse_schedule_payload(payload: str) -> dict[str, str]:
    info: dict[str, str] = {}
    for token in payload.split(","):
        token = token.strip()
        if not token or "=" not in token:
            continue
        key, value = token.split("=", 1)
        info[key.strip().lower()] = value.strip()
    return info


def _normalise_schedule(spec: Optional[str]) -> Optional[str]:
    if not spec:
        return None
    kind, sep, payload = spec.partition(":")
    if not sep:
        return spec
    kind = kind.strip().lower()
    payload = payload.strip()
    info = _parse_schedule_payload(payload)
    end_value = info.get("end") or info.get("final") or info.get("target")
    if end_value is None and kind == "linear" and payload:
        end_value = payload
    if end_value:
        try:
            numeric = float(end_value)
        except ValueError:
            numeric = None
        if numeric is not None:
            return f"linear:{numeric}"
    if kind == "linear" and payload:
        return f"linear:{payload}"
    return None


def _train_argv(args: argparse.Namespace) -> List[str]:
    dataset = _dataset_name(args)
    batch_size = _batch_size(args)
    run_id = _derive_run_id(args)
    output_dir = _derive_output_dir(args)
    precision = _coalesce(args.precision, args.teacher_precision, default="fp16")

    cli_args: List[str] = [
        f"--dataset={dataset}",
        f"--seq_len={args.block_size}",
        f"--batch={batch_size}",
        f"--alpha={args.alpha}",
        f"--temp={args.T}",
        f"--lr_base={args.lr_base}",
        f"--lr_peak={args.lr_peak}",
        f"--warmup_steps={args.warmup_steps}",
        f"--lr_scheduler={args.lr_scheduler}",
        f"--eval_every={args.eval_every}",
        f"--fallback_save_every={args.fallback_save_every}",
        f"--save_every_steps={args.save_every_steps}",
        f"--gcs_root={args.gcs_root or ''}",
        f"--run_id={run_id}",
        f"--phase_base={args.phase_base or 'phase'}",
        f"--phase_index={args.phase_index}",
        f"--teacher={args.teacher_name}",
        f"--teacher_precision={args.teacher_precision}",
        f"--precision={precision}",
        f"--max_steps={args.train_steps}",
        f"--improve_thresh={args.improve_thresh}",
        f"--output_dir={output_dir}",
        f"--seed={args.seed}",
    ]

    alpha_schedule = _normalise_schedule(args.alpha_schedule)
    if alpha_schedule:
        cli_args.append(f"--alpha_schedule={alpha_schedule}")
    elif args.alpha_schedule:
        print(
            "[trainer.entrypoint] Ignoring unsupported alpha_schedule spec:",
            args.alpha_schedule,
        )

    temp_schedule = _normalise_schedule(args.temp_schedule)
    if temp_schedule:
        cli_args.append(f"--temp_schedule={temp_schedule}")
    elif args.temp_schedule:
        print(
            "[trainer.entrypoint] Ignoring unsupported temp_schedule spec:",
            args.temp_schedule,
        )
    if args.save_every_seconds is not None:
        cli_args.append(f"--save_every_seconds={args.save_every_seconds}")

    return cli_args


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)

    train_args = _train_argv(args)
    if unknown:
        print("[trainer.entrypoint] Ignoring unused arguments:", " ".join(unknown))
    print("[trainer.entrypoint] Launching liquid_llm.train with:", " ".join(train_args))
    train_module.main(train_args)


if __name__ == "__main__":  # pragma: no cover
    main()
