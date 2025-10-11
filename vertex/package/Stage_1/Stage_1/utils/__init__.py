"""Utility helpers for Stage-1 package."""

from .config_utils import (
    Stage1Config,
    build_arg_parser,
    build_config,
    config_to_dict,
    ensure_output_path,
    dump_config,
    DEFAULT_DATASET_CFG,
)
from .schedule import WarmupCosineScheduler
from .io import open_sharded_file, path_exists, resolve_glob_paths, write_jsonl
from .secrets import get_secret, get_hf_token
from .ddp import setup_ddp, cleanup_ddp

__all__ = [
    "Stage1Config",
    "build_arg_parser",
    "build_config",
    "config_to_dict",
    "ensure_output_path",
    "dump_config",
    "DEFAULT_DATASET_CFG",
    "WarmupCosineScheduler",
    "open_sharded_file",
    "write_jsonl",
    "path_exists",
    "resolve_glob_paths",
    "get_secret",
    "get_hf_token",
    "setup_ddp",
    "cleanup_ddp",
]
