"""Concise logging utilities."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.io import write_jsonl


class StructuredLogger:
    """Console + JSONL logger for Stage-1 runs."""

    def __init__(self, output_path: Optional[str], run_id: Optional[str] = None, git_sha: Optional[str] = None):
        self.run_id = run_id or datetime.utcnow().strftime("stage1-%Y%m%d-%H%M%S")
        self.git_sha = git_sha
        self.output_path = output_path
        self.json_records: list[dict[str, Any]] = []
        self.log = logging.getLogger(f"Stage1Logger/{self.run_id}")
        self.log.setLevel(logging.INFO)
        if not self.log.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
            handler.setFormatter(formatter)
            self.log.addHandler(handler)

    def info(self, message: str, **fields: Any) -> None:
        payload = {"msg": message, **fields}
        payload.setdefault("run_id", self.run_id)
        if self.git_sha:
            payload.setdefault("git_sha", self.git_sha)
        self.log.info(json.dumps(payload, sort_keys=True))
        if fields:
            record = {"timestamp": datetime.utcnow().isoformat(), **payload}
            self.json_records.append(record)
            if self.output_path:
                self._flush()

    def metric(self, step: int, metrics: Dict[str, Any]) -> None:
        payload = {
            "run_id": self.run_id,
            "step": step,
            "metrics": metrics,
        }
        if self.git_sha:
            payload["git_sha"] = self.git_sha
        self.log.info(json.dumps(payload, sort_keys=True))
        if self.output_path:
            self.json_records.append({"timestamp": datetime.utcnow().isoformat(), **payload})
            self._flush()

    def health(self, status: str, detail: str, **fields: Any) -> None:
        payload = {"status": status, "detail": detail, **fields}
        self.info("health", **payload)

    def _flush(self) -> None:
        if not self.output_path:
            return
        path = os.path.join(self.output_path, "metrics.jsonl")
        write_jsonl(path, self.json_records)


__all__ = ["StructuredLogger"]
