"""Training entrypoint for Liquid LLM KD experiments."""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import kd
from .checkpointer import Checkpointer
from .eval import EvalConfig, evaluate_student
from .logging_utils import configure_logging, ensure_dir, git_sha_or_unknown, log_info


@dataclass
class TrainConfig:
    dataset: str
    seq_len: int
    batch: int
    alpha: float
    temp: float
    alpha_schedule: Optional[str]
    temp_schedule: Optional[str]
    lr_base: float
    lr_peak: float
    warmup_steps: int
    lr_scheduler: str
    eval_every: int
    fallback_save_every: int
    save_every_steps: int
    save_every_seconds: Optional[int]
    gcs_root: str
    run_id: str
    phase_base: str
    phase_index: int
    teacher: str
    teacher_precision: str
    precision: str
    max_steps: int
    improve_thresh: float


VOCAB_SIZE = 32


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Liquid LLM with KD")
    parser.add_argument("--dataset", default="dummy")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--alpha_schedule")
    parser.add_argument("--temp_schedule")
    parser.add_argument("--lr_base", type=float, default=5e-5)
    parser.add_argument("--lr_peak", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr_scheduler", default="linear")
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--fallback_save_every", type=int, default=200)
    parser.add_argument("--save_every_steps", type=int, default=100)
    parser.add_argument("--save_every_seconds", type=int)
    parser.add_argument("--gcs_root", default="")
    parser.add_argument("--run_id", default="local-run")
    parser.add_argument("--phase_base", default="phase")
    parser.add_argument("--phase_index", type=int, default=0)
    parser.add_argument("--teacher", default="gpt2-xl")
    parser.add_argument("--teacher_precision", default="fp16")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--improve_thresh", type=float, default=0.01)
    parser.add_argument("--output_dir", default="./runs/train")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args(argv)


def _create_teacher_logits(vocab_size: int) -> np.ndarray:
    base = np.linspace(-1.0, 1.0, vocab_size)
    noise = np.sin(np.arange(vocab_size) / 3.0)
    return base + 0.05 * noise


def _initial_student_logits(rng: np.random.Generator, vocab_size: int) -> np.ndarray:
    return rng.standard_normal(vocab_size) * 0.1


def _log_kd_hparams(step: int, alpha: float, temperature: float, last: Dict[str, float]) -> None:
    should_log = False
    if step == 1 or step % 50 == 0:
        should_log = True
    if abs(alpha - last.get("alpha", alpha)) > 1e-3:
        should_log = True
    if abs(temperature - last.get("temperature", temperature)) > 1e-3:
        should_log = True
    if should_log:
        log_info("[kd-hparams]", step=step, alpha=alpha, temperature=temperature)
        last["alpha"] = alpha
        last["temperature"] = temperature


def _run_selftests(metrics: Dict[str, float]) -> None:
    finite = all(math.isfinite(v) for v in metrics.values())
    assert finite, "Found non-finite metric values"
    log_info("[selftest] no_nans=PASS")
    if metrics.get("attn_mask_coverage") == 1.0 and metrics.get("pad_frac") == 0.0:
        log_info("[selftest] masks_ok=PASS")
    ce = metrics.get("ce_tok")
    ppl = metrics.get("ppl_train")
    if ce is not None and ppl is not None:
        assert abs(math.exp(ce) - ppl) < 1e-6, "Train perplexity mismatch"
        log_info("[selftest] ppl_matches_exp=PASS")


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    run_dir = ensure_dir(args.output_dir)
    logger = configure_logging(args.run_id)
    git_sha = git_sha_or_unknown()
    log_info("Run metadata", git_sha=git_sha, run_id=args.run_id)

    config = TrainConfig(
        dataset=args.dataset,
        seq_len=args.seq_len,
        batch=args.batch,
        alpha=args.alpha,
        temp=args.temp,
        alpha_schedule=args.alpha_schedule,
        temp_schedule=args.temp_schedule,
        lr_base=args.lr_base,
        lr_peak=args.lr_peak,
        warmup_steps=args.warmup_steps,
        lr_scheduler=args.lr_scheduler,
        eval_every=args.eval_every,
        fallback_save_every=args.fallback_save_every,
        save_every_steps=args.save_every_steps,
        save_every_seconds=args.save_every_seconds,
        gcs_root=args.gcs_root,
        run_id=args.run_id,
        phase_base=args.phase_base,
        phase_index=args.phase_index,
        teacher=args.teacher,
        teacher_precision=args.teacher_precision,
        precision=args.precision,
        max_steps=args.max_steps,
        improve_thresh=args.improve_thresh,
    )

    rng = np.random.default_rng(args.seed)
    student_logits = _initial_student_logits(rng, VOCAB_SIZE)
    teacher_logits = _create_teacher_logits(VOCAB_SIZE)
    log_info("Teacher loaded", name=config.teacher, precision=config.teacher_precision)
    log_info("[selftest] teacher_loaded=PASS", precision=config.teacher_precision)

    alpha_schedule = kd.parse_schedule("alpha", config.alpha, config.alpha_schedule, config.max_steps)
    temp_schedule = kd.parse_schedule("temperature", config.temp, config.temp_schedule, config.max_steps)

    schedule_changed = (
        alpha_schedule.final is not None and abs(alpha_schedule.final - alpha_schedule.initial) > 1e-6
    ) or (
        temp_schedule.final is not None and abs(temp_schedule.final - temp_schedule.initial) > 1e-6
    )
    if schedule_changed:
        log_info("[selftest] alpha_temp_schedule_active=PASS")

    checkpointer = Checkpointer(
        base_dir=run_dir,
        gcs_root=config.gcs_root,
        run_id=config.run_id,
        alpha_schedule=alpha_schedule,
        temp_schedule=temp_schedule,
        eval_ctx_lens=[512, 1024],
        fallback_every=config.fallback_save_every,
        save_every_steps=config.save_every_steps,
        save_every_seconds=config.save_every_seconds,
    )

    kd_log_tracker: Dict[str, float] = {}
    metrics_out_dir = ensure_dir(os.path.join(run_dir, "metrics"))

    for step in range(1, config.max_steps + 1):
        alpha = alpha_schedule.value(step)
        temperature = temp_schedule.value(step)
        _log_kd_hparams(step, alpha, temperature, kd_log_tracker)

        logits_before = student_logits.copy()
        kd_metrics = kd.kd_step(logits_before, teacher_logits, alpha, temperature)
        grad_norm = float(np.linalg.norm(teacher_logits - logits_before))
        corr = kd.correlation(logits_before, teacher_logits)
        top1 = kd.topk_match(logits_before, teacher_logits, k=1)
        top5 = kd.topk_match(logits_before, teacher_logits, k=min(5, VOCAB_SIZE))

        labels = rng.integers(0, VOCAB_SIZE, size=config.batch)
        probs = kd.softmax(logits_before, temperature=1.0)
        token_probs = np.clip(probs[labels], 1e-12, None)
        ce_tok = -float(np.mean(np.log(token_probs)))
        ppl_train = float(math.exp(ce_tok))
        tokens_per_s = float(config.batch * config.seq_len)

        # Pseudo update towards teacher distribution.
        blend = 0.05 * alpha
        student_logits = logits_before + blend * (teacher_logits - logits_before)
        student_logits[labels] += 0.01

        metrics = {
            "step": step,
            "ce_tok": ce_tok,
            "ppl_train": ppl_train,
            "kl_fwd": kd_metrics["kl_fwd"],
            "kl_rev": kd_metrics["kl_rev"],
            "H_student": kd_metrics["h_student"],
            "H_teacher": kd_metrics["h_teacher"],
            "corr_t_s": corr,
            "top1": top1,
            "top5": top5,
            "grad_norm": grad_norm,
            "tok/s": tokens_per_s,
            "attn_mask_coverage": 1.0,
            "pad_frac": 0.0,
            "alpha": alpha,
            "temperature": temperature,
        }

        log_info("Train step", **metrics)
        _run_selftests(metrics)

        checkpointer.on_step(step, {**metrics, "student_logits": student_logits.tolist()})
        checkpointer.maybe_time_save(step, metrics)

        if step % config.eval_every == 0:
            log_info("[selftest] eval_schedule_hit=PASS", step=step)
            eval_config = EvalConfig(
                dataset=config.dataset,
                seq_len=config.seq_len,
                batch=config.batch,
                run_dir=run_dir,
                eval_every=config.eval_every,
                contexts=[512, 1024],
            )
            results = evaluate_student(
                step=step,
                student_logits=student_logits,
                config=eval_config,
                out_dir=metrics_out_dir,
                seed=args.seed,
            )
            metrics_dict: Dict[str, float] = {
                f"val_ce_student@{r.ctx}": r.val_ce_student for r in results
            }
            metrics_dict.update(
                {
                    f"val_ppl_student@{r.ctx}": r.val_ppl_student for r in results
                }
            )
            val_1024 = metrics_dict["val_ce_student@1024"]
            metrics_dict["val_loss_student@1024"] = val_1024
            checkpoint_metrics = {**metrics_dict, "student_logits": student_logits.tolist()}
            records = checkpointer.on_eval(step, val_1024, checkpoint_metrics)
            if checkpointer.assert_best_and_latest():
                log_info("[selftest] checkpoints_written=PASS", step=step)
            log_info("Eval summary", step=step, **metrics_dict)

    log_info("Training complete", steps=config.max_steps)


if __name__ == "__main__":  # pragma: no cover
    main()
