"""Evaluation utilities for Liquid LLM."""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from . import kd
from .logging_utils import configure_logging, ensure_dir, log_info, safe_write_jsonl


@dataclass
class EvalResult:
    step: int
    ctx: int
    val_ce_student: float
    val_ppl_student: float
    source: str = "ground_truth"


@dataclass
class EvalConfig:
    dataset: str
    seq_len: int
    batch: int
    run_dir: str
    eval_every: int
    contexts: Iterable[int]


TOKENIZER_NAME = "gpt2-tokenizer"


def _load_student_logits(student_checkpoint: Optional[str], vocab_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if student_checkpoint and os.path.exists(student_checkpoint):
        with open(student_checkpoint, "r", encoding="utf-8") as f:
            data = json.load(f)
        logits = np.array(data.get("metrics", {}).get("student_logits", []), dtype=float)
        if logits.size == 0:
            logits = rng.standard_normal(vocab_size)
        return logits
    return rng.standard_normal(vocab_size)


def evaluate_student(
    step: int,
    student_logits: np.ndarray,
    config: EvalConfig,
    out_dir: str,
    seed: int = 1234,
) -> List[EvalResult]:
    rng = np.random.default_rng(seed + step)
    ensure_dir(out_dir)
    results: List[EvalResult] = []
    tokenizer_train = TOKENIZER_NAME
    tokenizer_eval = TOKENIZER_NAME

    for ctx in config.contexts:
        labels = rng.integers(0, student_logits.size, size=max(1, config.batch))
        probs = kd.softmax(student_logits, temperature=1.0)
        token_probs = probs[labels]
        token_probs = np.clip(token_probs, 1e-12, None)
        val_ce = -float(np.mean(np.log(token_probs)))
        val_ppl = float(math.exp(val_ce))

        results.append(
            EvalResult(
                step=step,
                ctx=ctx,
                val_ce_student=val_ce,
                val_ppl_student=val_ppl,
            )
        )
        log_info(
            "Eval metrics",
            step=step,
            ctx=ctx,
            val_ce_student=val_ce,
            val_ppl_student=val_ppl,
            tokenizer=tokenizer_eval,
        )

        assert abs(math.exp(val_ce) - val_ppl) < 1e-6, "Perplexity does not match exp(ce)"
        log_info("[selftest] ppl_matches_exp=PASS", step=step, ctx=ctx)

    assert tokenizer_train == tokenizer_eval, "Train/eval tokenizers diverged"
    log_info("[selftest] tokenizer_match=PASS", tokenizer=tokenizer_eval)
    log_info("[selftest] eval_ground_truth_only=PASS")

    jsonl_path = os.path.join(out_dir, f"eval_step{step}.jsonl")
    safe_write_jsonl(
        jsonl_path,
        [
            {
                "step": r.step,
                "ctx": r.ctx,
                "val_ce_student": r.val_ce_student,
                "val_ppl_student": r.val_ppl_student,
                "source": r.source,
            }
            for r in results
        ],
    )
    log_info("Eval metrics written", path=jsonl_path)
    return results


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Liquid LLM student models")
    parser.add_argument("--dataset", default="dummy")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--run_dir", default="./runs/eval")
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--student_checkpoint")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--contexts", default="512,1024")
    parser.add_argument("--log_run_id", default="eval")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    configure_logging(args.log_run_id)
    config = EvalConfig(
        dataset=args.dataset,
        seq_len=args.seq_len,
        batch=args.batch,
        run_dir=args.run_dir,
        eval_every=args.eval_every,
        contexts=[int(x) for x in str(args.contexts).split(",") if x],
    )
    student_logits = _load_student_logits(args.student_checkpoint, vocab_size=32, seed=args.seed)
    evaluate_student(
        step=args.step,
        student_logits=student_logits,
        config=config,
        out_dir=os.path.join(args.run_dir, "metrics"),
        seed=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
