"""Command line interface for Vertex AI custom training."""
from __future__ import annotations

import argparse
import os
from typing import Optional
from torch.utils.data import DataLoader

from . import data
from .gcs_io import ensure_local_dir
from .model_init import initialize_student
from .runtime_setup import install_flash_attn_from_gcs, login_hf
from .teacher import TeacherConfig, TeacherWrapper
from .train import Trainer
from .utils import AnnealingSchedule, configure_logging, detect_training_device, json_log, set_seed

DEFAULTS = {
    "resume_gcs_uri": "gs://liquid-llm-bucket-2/stage1/stage1.pt",
    "output_gcs_uri": "gs://liquid-llm-bucket-2/stage1/Checkpoints/vertex-runs",
    "teacher_id": "meta-llama/Llama-3.2-3B",
    "teacher_mode": "precompute",
    "seq_len": 1024,
    "dataset_manifest": "gs://liquid-llm-bucket-2/datasets/stage1/manifests/stage1.jsonl",
    "tool_use_ratio": 0.08,
    "kd_temperature": 2.0,
    "kd_alpha_start": 0.7,
    "kd_alpha_end": 0.4,
    "kd_anneal_pct": 0.3,
    "keep_old_logit_l2": 0.1,
    "keep_old_logit_l2_fade_step": 30000,
    "precision": "bfloat16",
    "lr": 2.5e-4,
    "weight_decay": 0.1,
    "betas": "0.9,0.95",
    "warmup_steps": 3000,
    "max_steps": 120000,
    "eval_every": 1000,
    "save_every": 2000,
    "fa_wheel_gcs": "gs://liquid-llm-bucket-2/FlashAttention/flash_attn-2.8.3+cu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl",
    "logs_gcs_uri": "gs://liquid-llm-bucket-2/logs/stage1_console/",
    "tb_gcs_uri": "gs://liquid-llm-bucket-2/logs/tb/",
    "teacher_logits_dir": "gs://liquid-llm-bucket-2/teacher/llama-3.2-3b/logits",
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Liquid LLM Stage-1 Trainer")
    for key, value in DEFAULTS.items():
        arg = f"--{key}".replace("_", "-")
        parser.add_argument(arg, default=value)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-mode", choices=["precompute", "online"], default=DEFAULTS["teacher_mode"])
    parser.add_argument("--precision", choices=["bfloat16", "fp16", "fp32"], default=DEFAULTS["precision"])
    parser.add_argument("--max-steps", type=int, default=DEFAULTS["max_steps"])
    parser.add_argument("--eval-every", type=int, default=DEFAULTS["eval_every"])
    parser.add_argument("--save-every", type=int, default=DEFAULTS["save_every"])
    parser.add_argument("--warmup-steps", type=int, default=DEFAULTS["warmup_steps"])
    parser.add_argument("--keep-old-logit-l2-fade-step", type=int, default=DEFAULTS["keep_old_logit_l2_fade_step"])
    parser.add_argument("--seq-len", type=int, default=DEFAULTS["seq_len"])
    parser.add_argument("--kd-temperature", type=float, default=DEFAULTS["kd_temperature"])
    parser.add_argument("--kd-alpha-start", type=float, default=DEFAULTS["kd_alpha_start"])
    parser.add_argument("--kd-alpha-end", type=float, default=DEFAULTS["kd_alpha_end"])
    parser.add_argument("--kd-anneal-pct", type=float, default=DEFAULTS["kd_anneal_pct"])
    parser.add_argument("--keep-old-logit-l2", type=float, default=DEFAULTS["keep_old_logit_l2"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--tool-use-ratio", type=float, default=DEFAULTS["tool_use_ratio"])
    parser.add_argument("--batch-per-device", type=int, default=1)
    return parser.parse_args(argv)


def _parse_betas(betas: str) -> tuple[float, float]:
    parts = [float(x) for x in betas.split(",")]
    if len(parts) != 2:
        raise ValueError("Betas must contain two comma separated values")
    return parts[0], parts[1]


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    logger = configure_logging()
    set_seed(args.seed)
    login_hf()
    install_flash_attn_from_gcs(args.fa_wheel_gcs)
    device_info = detect_training_device()
    logger.info(
        "Stage-1 KD starting | device=%s | capability=%s | seq_len=%d | teacher_mode=%s",
        device_info.name,
        device_info.capability,
        args.seq_len,
        args.teacher_mode,
    )
    tokenizer = data.build_tokenizer(args.teacher_id)
    manifest_entries = data.read_manifest(args.dataset_manifest)
    dataset = data.ManifestDataset(manifest_entries, tokenizer, seq_len=args.seq_len, tool_use_ratio=args.tool_use_ratio)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = None
    run_id = os.environ.get("AIP_TRAINING_JOB_ID", "local")
    local_root = ensure_local_dir(os.path.join("/tmp", "vertex_run", run_id))
    model = initialize_student(args.resume_gcs_uri, local_root, seq_len=args.seq_len)
    output_gcs = args.output_gcs_uri.rstrip("/") + f"/{run_id}"
    betas = _parse_betas(args.betas)
    kd_alpha_schedule = AnnealingSchedule(args.kd_alpha_start, args.kd_alpha_end, args.kd_anneal_pct)
    ce_beta_schedule = AnnealingSchedule(1 - args.kd_alpha_start, 1 - args.kd_alpha_end, args.kd_anneal_pct)
    logit_l2_schedule = AnnealingSchedule(args.keep_old_logit_l2, 0.0, args.keep_old_logit_l2_fade_step / max(1, args.max_steps))
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device_info.device,
        output_dir=local_root,
        output_gcs_uri=output_gcs,
        lr=args.lr,
        betas=betas,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        kd_temperature=args.kd_temperature,
        kd_alpha_schedule=kd_alpha_schedule,
        ce_beta_schedule=ce_beta_schedule,
        logit_l2_gamma_schedule=logit_l2_schedule,
        logit_reference=None,
        precision=args.precision,
    )
    json_log(
        logger,
        {
            "teacher_mode": args.teacher_mode,
            "dataset_size": len(dataset),
            "seq_len": args.seq_len,
            "output_gcs": output_gcs,
        },
    )
    if args.teacher_mode == "online":
        teacher = TeacherWrapper(TeacherConfig(model_id=args.teacher_id))
        # Example integration: pre-fetch logits for first batch
        for batch in train_loader:
            batch["teacher_logits"] = teacher.logits(batch["input_ids"].to(device_info.device)).cpu()
            break
    trainer.train()


if __name__ == "__main__":
    main()
