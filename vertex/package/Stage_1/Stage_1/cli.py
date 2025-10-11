"""CLI for Stage-1 training."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from .trainer import Stage1Trainer
from .utils import build_arg_parser, build_config, dump_config, ensure_output_path, path_exists


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = build_config(args)
    if not config.resume_gcs_uri:
        raise SystemExit("Stage-1 requires --resume_gcs_uri=gs://.../stage1.pt")
    if not path_exists(config.resume_gcs_uri):
        raise SystemExit(f"Checkpoint not found at {config.resume_gcs_uri}")
    if not config.run_id:
        config.run_id = datetime.now(timezone.utc).strftime("stage1-%Y%m%d-%H%M%S")
    if config.output_gcs_uri:
        base_uri = config.output_gcs_uri.rstrip("/")
        config.output_gcs_uri = f"{base_uri}/{config.run_id}"
    if config.output_gcs_uri and not config.output_gcs_uri.startswith("gs://"):
        ensure_output_path(config.output_gcs_uri)
    trainer = Stage1Trainer(config)
    config_path = os.path.join(trainer.output_path, "config_stage1.json")
    dump_config(config_path, config)
    trainer.train()


if __name__ == "__main__":  # pragma: no cover
    main()
