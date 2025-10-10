"""CLI for Stage-1 training."""

from __future__ import annotations

import json
import os
import uuid

from .trainer import Stage1Trainer
from .utils import build_arg_parser, build_config, dump_config, ensure_output_path


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = build_config(args)
    if not config.run_id:
        config.run_id = str(uuid.uuid4())
    if config.output_gcs_uri and not config.output_gcs_uri.startswith("gs://"):
        ensure_output_path(config.output_gcs_uri)
    trainer = Stage1Trainer(config)
    config_path = os.path.join(trainer.output_path, "config_stage1.json")
    dump_config(config_path, config)
    trainer.train()


if __name__ == "__main__":  # pragma: no cover
    main()
