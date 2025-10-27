"""Vertex AI entrypoint."""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch

from . import runtime_setup
from .data import DataModule
from .model_init import load_student_model
from .teacher import load_teacher, load_teacher_tokenizer, precompute_teacher_logits
from .train import TeacherProvider, TrainingConfig, train
from .utils import parse_betas, set_seed, str2bool

DEFAULT_FA_WHEEL = "gs://liquid-llm-bucket-2/FlashAttention/flash_attn-2.8.3+cu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
DEFAULT_RESUME = "gs://liquid-llm-bucket-2/stage1/stage1.pt"
DEFAULT_MANIFEST = "gs://liquid-llm-bucket-2/datasets/stage1/manifests/stage1.jsonl"
DEFAULT_OUTPUT = "gs://liquid-llm-bucket-2/stage1/Checkpoints/vertex-runs"
DEFAULT_LOGITS_DIR = "gs://liquid-llm-bucket-2/teacher/llama-3.2-3b/logits/"

LOGGER = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Liquid LLM Stage 1 Vertex trainer")
    parser.add_argument("--mode", choices=["train", "precompute_teacher_logits"], default="train")
    parser.add_argument("--resume_gcs_uri", default=DEFAULT_RESUME)
    parser.add_argument("--output_gcs_uri", default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset_manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--teacher_id", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--teacher_mode", choices=["online", "precompute"], default="precompute")
    parser.add_argument("--fa_wheel_gcs", default=DEFAULT_FA_WHEEL)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--precision", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--betas", type=parse_betas, default=(0.9, 0.95))
    parser.add_argument("--warmup_steps", type=int, default=3000)
    parser.add_argument("--max_steps", type=int, default=120000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--kd_alpha_start", type=float, default=0.7)
    parser.add_argument("--kd_alpha_end", type=float, default=0.4)
    parser.add_argument("--kd_anneal_pct", type=float, default=0.3)
    parser.add_argument("--keep_old_logit_l2", type=float, default=0.1)
    parser.add_argument("--keep_old_logit_l2_fade_step", type=int, default=30000)
    parser.add_argument("--tool_use_ratio", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", default="/tmp/logs")
    parser.add_argument("--tb_dir", default="/tmp/tensorboard")
    parser.add_argument("--tb_gcs_uri", default="gs://liquid-llm-bucket-2/logs/tb/")
    parser.add_argument("--logits_dir", default=DEFAULT_LOGITS_DIR)
    parser.add_argument("--cache_dir", default="/cache/hf")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--teacher_batch_size", type=int, default=4)
    parser.add_argument("--teacher_seq_len", type=int, default=1024)
    parser.add_argument("--teacher_precision", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--base_model_id", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--width_scale", type=float, default=1.2)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    set_seed(args.seed)

    if args.mode == "precompute_teacher_logits":
        dtype = torch.bfloat16 if args.teacher_precision == "bfloat16" else torch.float16
        precompute_teacher_logits(
            manifest_uri=args.dataset_manifest,
            output_dir=args.logits_dir,
            model_id=args.teacher_id,
            dtype=dtype,
            batch_size=args.teacher_batch_size,
            seq_len=args.teacher_seq_len,
        )
        return

    dtype = runtime_setup.setup_runtime(args.fa_wheel_gcs, precision=args.precision)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_student_model(
        resume_gcs_uri=args.resume_gcs_uri,
        device=device,
        precision=args.precision,
        base_model_id=args.base_model_id,
        width_scale=args.width_scale,
    )

    teacher_model = None
    if args.teacher_mode == "online":
        teacher_model, tokenizer = load_teacher(model_id=args.teacher_id, dtype=dtype, cache_dir=args.cache_dir)
    else:
        tokenizer = load_teacher_tokenizer(model_id=args.teacher_id, cache_dir=args.cache_dir)

    datamodule = DataModule(
        manifest_uri=args.dataset_manifest,
        tokenizer=tokenizer,
        block_size=args.block_size,
        batch_size=args.batch_size,
        seed=args.seed,
        tool_use_ratio=args.tool_use_ratio,
    )

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    log_path = Path(args.log_dir) / f"{run_id}.log"
    tb_path = Path(args.tb_dir) / run_id

    config = TrainingConfig(
        learning_rate=args.lr,
        betas=args.betas,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        kd_temperature=args.kd_temperature,
        kd_alpha_start=args.kd_alpha_start,
        kd_alpha_end=args.kd_alpha_end,
        kd_anneal_pct=args.kd_anneal_pct,
        keep_old_logit_l2=args.keep_old_logit_l2,
        keep_old_logit_l2_fade_step=args.keep_old_logit_l2_fade_step,
        output_gcs_uri=args.output_gcs_uri,
        run_id=run_id,
        precision=args.precision,
    )

    teacher_provider = TeacherProvider(
        mode=args.teacher_mode,
        logits_dir=args.logits_dir,
        teacher_model=teacher_model,
    )

    train(
        model=model,
        datamodule=datamodule,
        device=device,
        teacher_provider=teacher_provider,
        config=config,
        precision_dtype=dtype,
        log_path=log_path,
        tensorboard_dir=tb_path,
    )

    gcs_target = f"{args.tb_gcs_uri.rstrip('/')}/{run_id}"
    try:
        from . import gcs_io

        gcs_io.gcs_cp(str(tb_path), gcs_target, recursive=True)
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.warning("Failed to upload TensorBoard logs: %s", exc)


if __name__ == "__main__":
    main(sys.argv[1:])
