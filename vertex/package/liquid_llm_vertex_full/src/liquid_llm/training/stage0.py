"""Command line interface for the Vertex Stage0 trainer."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, MutableMapping, Optional

import yaml

# Default hyper-parameter values mirror the public Stage0 configuration.
_DEFAULTS: Dict[str, Any] = {
    "resume_gcs_uri": None,
    "block_size": 512,
    "teacher_name": "gpt2-xl",
    "dataset_name": "wikitext",
    "dataset_config": "wikitext-103-raw-v1",
    "output_gcs_uri": None,
    "local_workdir": "/tmp/liquid_work",
    "seed": 42,
    "global_batch": 64,
    "micro_batch": 8,
    "lr": 3e-4,
    "weight_decay": 0.1,
    "betas": [0.9, 0.95],
    "eps": 1e-8,
    "warmup_steps": 2000,
    "train_steps": 45_000,
    "eval_every": 500,
    "save_every": 1000,
    "log_interval": 50,
    "grad_clip": 1.0,
    "kd_alpha": 0.5,
    "kd_temperature": 1.0,
    "time_ckpt_secs": 1800,
    "time_ckpt_retention_secs": 14_400,
    "time_ckpt_keep_k": None,
    "best_ckpt_keep_k": 3,
    "best_ckpt_retention_secs": None,
    "step_ckpt_keep_k": 5,
    "step_ckpt_retention_secs": None,
    "precision": "no",
    "model": {
        "d_model": 768,
        "n_layers": 10,
        "n_heads": 12,
        "dropout": 0.0,
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch Liquid-LLM Stage0 training on Vertex"
    )

    # Core dataset + IO arguments (Vertex wires resume/output paths).
    parser.add_argument("--resume_gcs_uri", type=str, default=None)
    parser.add_argument("--block_size", type=int, required=True)
    parser.add_argument("--teacher_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, required=True)

    parser.add_argument("--output_gcs_uri", type=str, default=None)
    parser.add_argument("--local_workdir", type=str, default=_DEFAULTS["local_workdir"])

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML file containing default configuration overrides.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--global_batch", type=int, default=None)
    parser.add_argument("--micro_batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--betas", type=float, nargs=2, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--train_steps", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--teacher_eval_every", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--kd_alpha", type=float, default=None)
    parser.add_argument("--kd_temperature", type=float, default=None)

    # Vertex-managed checkpoint controls.
    parser.add_argument("--time_ckpt_secs", type=int, default=None)
    parser.add_argument("--time_ckpt_retention_secs", type=int, default=None)
    parser.add_argument("--time_ckpt_keep_k", type=int, default=None)
    parser.add_argument("--best_ckpt_keep_k", type=int, default=None)
    parser.add_argument("--best_ckpt_retention_secs", type=int, default=None)
    parser.add_argument("--step_ckpt_keep_k", type=int, default=None)
    parser.add_argument("--step_ckpt_retention_secs", type=int, default=None)

    # Precision knobs (Vertex passes bf16/fp16 flags separately).
    parser.add_argument("--fp16", action="store_true", help="Force fp16 training")
    parser.add_argument("--bf16", action="store_true", help="Force bf16 training")

    # Model architecture overrides.
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    return parser


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    data = yaml.safe_load(Path(path).read_text())
    if data is None:
        return {}
    if not isinstance(data, MutableMapping):
        raise ValueError("Trainer config YAML must contain a mapping at the top level")
    return dict(data)


def _cli_override(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def _ensure_jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, MutableMapping):
        return {k: _ensure_jsonable(v) for k, v in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        result = []
        for item in value:
            if isinstance(item, MutableMapping) or (
                isinstance(item, Iterable) and not isinstance(item, (str, bytes))
            ):
                result.append(_ensure_jsonable(item))
            else:
                result.append(item)
        return result
    return value


def parse_args(argv: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    parser = _parser()
    args = parser.parse_args(argv)

    merged = deepcopy(_DEFAULTS)
    yaml_overrides = _load_yaml(args.config)
    for key, value in yaml_overrides.items():
        if key == "model" and isinstance(value, MutableMapping):
            merged["model"].update(value)
        else:
            merged[key] = value

    def choose(name: str, default: Any = None) -> Any:
        cli_value = getattr(args, name)
        fallback = merged.get(name, default)
        return _cli_override(cli_value, fallback)

    betas = choose("betas")
    if betas is not None:
        betas = list(betas)

    precision = merged.get("precision", _DEFAULTS["precision"])
    if args.bf16:
        precision = "bf16"
    elif args.fp16:
        precision = "fp16"

    teacher_eval_every = choose("teacher_eval_every")
    eval_every = choose("eval_every")
    if teacher_eval_every is not None:
        eval_every = teacher_eval_every

    cfg: Dict[str, Any] = {
        "resume_gcs_uri": args.resume_gcs_uri,
        "block_size": args.block_size,
        "teacher_name": args.teacher_name,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "output_gcs_uri": args.output_gcs_uri,
        "local_workdir": args.local_workdir,
        "seed": choose("seed"),
        "global_batch": choose("global_batch"),
        "micro_batch": choose("micro_batch"),
        "lr": choose("lr"),
        "weight_decay": choose("weight_decay"),
        "betas": betas,
        "eps": choose("eps"),
        "warmup_steps": choose("warmup_steps"),
        "train_steps": choose("train_steps"),
        "eval_every": eval_every,
        "save_every": choose("save_every"),
        "log_interval": choose("log_interval"),
        "grad_clip": choose("grad_clip"),
        "kd_alpha": choose("kd_alpha"),
        "kd_temperature": choose("kd_temperature"),
        "time_ckpt_secs": choose("time_ckpt_secs"),
        "time_ckpt_retention_secs": choose("time_ckpt_retention_secs"),
        "time_ckpt_keep_k": choose("time_ckpt_keep_k"),
        "best_ckpt_keep_k": choose("best_ckpt_keep_k"),
        "best_ckpt_retention_secs": choose("best_ckpt_retention_secs"),
        "step_ckpt_keep_k": choose("step_ckpt_keep_k"),
        "step_ckpt_retention_secs": choose("step_ckpt_retention_secs"),
        "precision": precision,
        "model": {
            "d_model": _cli_override(args.d_model, merged["model"].get("d_model")),
            "n_layers": _cli_override(args.n_layers, merged["model"].get("n_layers")),
            "n_heads": _cli_override(args.n_heads, merged["model"].get("n_heads")),
            "dropout": _cli_override(args.dropout, merged["model"].get("dropout")),
        },
    }

    if teacher_eval_every is not None:
        cfg["teacher_eval_every"] = teacher_eval_every

    return _ensure_jsonable(cfg)


__all__ = ["parse_args"]