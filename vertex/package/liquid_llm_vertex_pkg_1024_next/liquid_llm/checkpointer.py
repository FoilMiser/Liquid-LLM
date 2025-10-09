"""Checkpoint management for structured experiments."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import kd
from .logging_utils import ensure_dir, log_info


@dataclass
class CheckpointRecord:
    path: str
    kind: str
    step: int
    metric: float
    meta: Dict[str, object] = field(default_factory=dict)


class Checkpointer:
    """Handles various checkpoint saving policies."""

    def __init__(
        self,
        base_dir: str,
        gcs_root: str,
        run_id: str,
        alpha_schedule: kd.Schedule,
        temp_schedule: kd.Schedule,
        eval_ctx_lens: List[int],
        fallback_every: int,
        save_every_steps: int,
        save_every_seconds: Optional[int] = None,
    ) -> None:
        self.base_dir = ensure_dir(os.path.join(base_dir, "checkpoints"))
        self.gcs_root = gcs_root
        self.run_id = run_id
        self.alpha_schedule = alpha_schedule
        self.temp_schedule = temp_schedule
        self.eval_ctx_lens = eval_ctx_lens
        self.fallback_every = max(1, fallback_every)
        self.save_every_steps = max(1, save_every_steps)
        self.save_every_seconds = save_every_seconds

        self.best_metric = float("inf")
        self.best_record: Optional[CheckpointRecord] = None
        self.latest_record: Optional[CheckpointRecord] = None
        self.last_fallback_step = 0
        self.last_step_save = 0
        self.last_time_save = time.time()

    # Internal helpers -----------------------------------------------------
    def _target_path(self, kind: str, step: int, metric: float) -> str:
        if kind == "step":
            return os.path.join(self.base_dir, f"step_{step:06d}.pt")
        if kind == "fallback":
            return os.path.join(self.base_dir, f"fallback_{step:06d}.pt")
        if kind == "best":
            return os.path.join(self.base_dir, "best.pt")
        if kind == "latest":
            return os.path.join(self.base_dir, "latest.pt")
        if kind == "time":
            return os.path.join(self.base_dir, f"time_{int(time.time())}.pt")
        if kind == "versioned_best":
            return os.path.join(
                self.base_dir,
                f"ckpt_best_step{step}_vc{metric:.4f}.pt",
            )
        raise ValueError(f"Unknown checkpoint kind: {kind}")

    def _write_checkpoint(self, kind: str, step: int, metric: float, metrics: Dict[str, float]) -> CheckpointRecord:
        path = self._target_path(kind, step, metric)
        ensure_dir(os.path.dirname(path))
        payload = {
            "step": step,
            "metric": metric,
            "metrics": metrics,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
        meta = {
            "kind": kind,
            "path": path,
            "gcs_uri": f"{self.gcs_root}/{self.run_id}/{os.path.basename(path)}" if self.gcs_root else path,
            "schedule": json.loads(kd.schedule_to_json(self.alpha_schedule, self.temp_schedule)),
            "eval_ctx_lens": self.eval_ctx_lens,
        }
        log_info("[ckpt-meta]", **meta)
        return CheckpointRecord(path=path, kind=kind, step=step, metric=metric, meta=meta)

    # Public API -----------------------------------------------------------
    def on_step(self, step: int, metrics: Dict[str, float]) -> Optional[CheckpointRecord]:
        if step - self.last_step_save >= self.save_every_steps:
            self.last_step_save = step
            return self._write_checkpoint("step", step, metrics.get("ce_tok", 0.0), metrics)
        return None

    def maybe_time_save(self, step: int, metrics: Dict[str, float]) -> Optional[CheckpointRecord]:
        if self.save_every_seconds is None:
            return None
        now = time.time()
        if now - self.last_time_save >= self.save_every_seconds:
            self.last_time_save = now
            return self._write_checkpoint("time", step, metrics.get("ce_tok", 0.0), metrics)
        return None

    def on_eval(self, step: int, student_val_ce: float, metrics: Dict[str, float]) -> Dict[str, Optional[CheckpointRecord]]:
        records: Dict[str, Optional[CheckpointRecord]] = {
            "best": None,
            "latest": None,
            "fallback": None,
            "versioned": None,
        }

        if student_val_ce < self.best_metric:
            self.best_metric = student_val_ce
            records["best"] = self._write_checkpoint("best", step, student_val_ce, metrics)
            records["versioned"] = self._write_checkpoint("versioned_best", step, student_val_ce, metrics)
            self.best_record = records["best"]

        records["latest"] = self._write_checkpoint("latest", step, student_val_ce, metrics)
        self.latest_record = records["latest"]

        if step - self.last_fallback_step >= self.fallback_every:
            self.last_fallback_step = step
            records["fallback"] = self._write_checkpoint("fallback", step, student_val_ce, metrics)

        return records

    def assert_best_and_latest(self) -> bool:
        return self.best_record is not None and self.latest_record is not None
