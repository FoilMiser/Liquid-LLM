"""Utilities for loading configuration files and merging CLI overrides."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import yaml

from .io import open_sharded_file


DEFAULT_DATASET_CFG = "gs://liquid-llm-bucket-2/datasets/stage1.jsonl"


@dataclass
class Stage1Config:
    """In-memory representation of Stage-1 experiment configuration."""

    resume_gcs_uri: Optional[str] = None
    output_gcs_uri: Optional[str] = None
    run_id: Optional[str] = None
    teacher_name: str = "meta-llama/Meta-Llama-3.1-8B"
    teacher_endpoint: Optional[str] = None
    teacher_max_batch_size: int = 0
    dataset_cfg: Optional[str] = DEFAULT_DATASET_CFG
    seq_len: int = 1024
    block_size: int = 1024
    train_steps: int = 250_000
    batch_size: int = 8
    eval_batch_size: int = 8
    throughput_tokens: int = 32_768
    use_flash_attn: bool = True
    use_grad_ckpt: bool = True
    dtype: str = "bfloat16"
    device: str = "cuda"
    optimizer: str = "adamw"
    lr: float = 2.5e-4
    weight_decay: float = 0.1
    betas: str = "0.9,0.95"
    warmup_steps: int = 3000
    eval_every: int = 1000
    save_every: int = 2000
    log_every: int = 100
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    hf_secret_name: Optional[str] = "hf_token"
    hf_cache_dir: Optional[str] = None
    seed: int = 42
    tokenizer_name: Optional[str] = None
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 15
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    kd_temperature: float = 2.0
    kd_alpha_start: float = 0.7
    kd_alpha_end: float = 0.4
    kd_anneal_pct: float = 0.3
    keep_old_logit_l2: float = 0.1
    keep_old_logit_l2_fade_step: int = 30_000
    keep_old_logit_l2_enable: bool = True
    tool_use_ratio: float = 0.08
    calculator_enabled: bool = True
    scratchpad_enabled: bool = True
    use_checkpoint_saver: bool = True
    best_metric: str = "val_perplexity"
    best_metric_mode: str = "min"
    num_workers: int = 4
    extra: Dict[str, Any] = field(default_factory=dict)


def _coerce_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return value


def update_from_mapping(config: MutableMapping[str, Any], updates: Mapping[str, Any]) -> None:
    """Recursively update config dictionary with values from updates."""

    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(config.get(key), MutableMapping):
            update_from_mapping(config[key], value)
        else:
            config[key] = value


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(raw)!r}")
    return raw


def parse_kv_overrides(overrides: Iterable[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key, value = override.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in {"true", "false"}:
            parsed: Any = value.lower() == "true"
        else:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = value
        result[key] = parsed
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-1 training entrypoint")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--config_override",
        type=str,
        nargs="*",
        default=None,
        help="Additional KEY=VALUE overrides",
    )
    parser.add_argument("--resume_gcs_uri", type=str, required=False)
    parser.add_argument("--output_gcs_uri", type=str, required=False)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--teacher_name", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--teacher_endpoint", type=str, default=None)
    parser.add_argument("--teacher_max_batch_size", type=int, default=0)
    parser.add_argument("--dataset_cfg", type=str, default=DEFAULT_DATASET_CFG)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--train_steps", type=int, default=250_000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--throughput_tokens", type=int, default=32_768)
    parser.add_argument("--use_flash_attn", type=_coerce_bool, default=True)
    parser.add_argument("--use_grad_ckpt", type=_coerce_bool, default=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--betas", type=str, default="0.9,0.95")
    parser.add_argument("--warmup_steps", type=int, default=3000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--hf_secret_name", type=str, default="hf_token")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--layer_norm_eps", type=float, default=1e-5)
    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--kd_alpha_start", type=float, default=0.7)
    parser.add_argument("--kd_alpha_end", type=float, default=0.4)
    parser.add_argument("--kd_anneal_pct", type=float, default=0.3)
    parser.add_argument("--keep_old_logit_l2", type=float, default=0.1)
    parser.add_argument("--keep_old_logit_l2_fade_step", type=int, default=30_000)
    parser.add_argument("--keep_old_logit_l2_enable", type=_coerce_bool, default=True)
    parser.add_argument("--tool_use_ratio", type=float, default=0.08)
    parser.add_argument("--calculator_enabled", type=_coerce_bool, default=True)
    parser.add_argument("--scratchpad_enabled", type=_coerce_bool, default=True)
    parser.add_argument("--use_checkpoint_saver", type=_coerce_bool, default=True)
    parser.add_argument("--best_metric", type=str, default="val_perplexity")
    parser.add_argument("--best_metric_mode", type=str, default="min")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser


def _asdict(config: Stage1Config) -> Dict[str, Any]:
    base = config.__dict__.copy()
    base.update(base.pop("extra", {}))
    return base


def build_config(args: argparse.Namespace) -> Stage1Config:
    yaml_cfg = load_yaml_config(getattr(args, "config", None))
    cfg_dict = {**Stage1Config().__dict__, **yaml_cfg}
    cli_overrides = {
        key: getattr(args, key)
        for key in cfg_dict
        if hasattr(args, key) and getattr(args, key) is not None
    }
    cfg_dict.update(cli_overrides)
    if args.config_override:
        cfg_dict.update(parse_kv_overrides(args.config_override))
    extra_keys = [key for key in cfg_dict.keys() if not hasattr(Stage1Config, key)]
    extra: Dict[str, Any] = {}
    for key in extra_keys:
        extra[key] = cfg_dict.pop(key)
    cfg = Stage1Config(**cfg_dict)
    if not cfg.tokenizer_name:
        cfg.tokenizer_name = cfg.teacher_name
    cfg.dtype = str(cfg.dtype).lower()
    cfg.extra = extra
    return cfg


def config_to_dict(config: Stage1Config) -> Dict[str, Any]:
    return _asdict(config)


def ensure_output_path(path: str) -> None:
    if path.startswith("gs://"):
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def dump_config(path: str, config: Stage1Config) -> None:
    ensure_output_path(os.path.dirname(path) or ".")
    with open_sharded_file(path, "w") as f:
        json.dump(config_to_dict(config), f, indent=2, sort_keys=True)


__all__ = [
    "Stage1Config",
    "build_arg_parser",
    "build_config",
    "config_to_dict",
    "ensure_output_path",
    "dump_config",
    "DEFAULT_DATASET_CFG",
]
