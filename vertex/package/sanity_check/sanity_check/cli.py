"""Command-line entrypoint for the sanity check runner."""

from __future__ import annotations

import argparse
import logging
import sys

import torch

from .runner import DEFAULT_GCS_URI, run


def _default_dtype() -> str:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float32"


def _default_device() -> str:
    """Select an appropriate default torch device."""

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-0 checkpoint sanity checks")
    parser.add_argument("--checkpoint_gcs_uri", type=str, default=DEFAULT_GCS_URI, help="Checkpoint URI on GCS or local path")
    parser.add_argument(
        "--device",
        type=str,
        default=_default_device(),
        help="Torch device to run tests on (defaults to CUDA when available, otherwise CPU)",
    )
    parser.add_argument("--block_size", type=int, default=512, help="Model block size / sequence length")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Tokenizer name or path")
    parser.add_argument("--throughput_tokens", type=int, default=32768, help="Total tokens to process during throughput probe")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size for smoke tests")
    parser.add_argument("--dtype", type=str, choices=["float32", "bfloat16", "float16"], default=_default_dtype(), help="Computation dtype")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_time_s", type=float, default=180.0, help="Maximum allowed runtime in seconds")
    parser.add_argument("--log_level", type=str, default="INFO", help="Python logging level")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    try:
        exit_code = run(args)
    except Exception as exc:  # pragma: no cover
        logging.exception("Sanity check run failed")
        sys.exit(1)
    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
