"""Command line interface for Vertex AI custom training."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

import yaml
from torch.utils.data import DataLoader

from . import data
from .gcs_io import ensure_local_dir, local_to_gcs
from .model_init import apply_grad_checkpointing, initialize_student
from .prep import DatasetSpec, ensure_toolkit, load_datasets_yaml, normalize_gcs_uri, prepare_if_needed
from .runtime_setup import enable_flash_attn_if_available, install_flash_attn_from_gcs, login_hf
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
    "grad_checkpoint": True,
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Liquid LLM Stage-1 Trainer", conflict_handler="resolve")
    for key, value in DEFAULTS.items():
        arg = f"--{key}".replace("_", "-")
        parser.add_argument(arg, default=value)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
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
    parser.add_argument("--prepare-data", choices=["auto", "force", "skip"], default="auto")
    parser.add_argument(
        "--prep-toolkit-zip-uri",
        default="gs://liquid-llm-bucket-2/sandbox/preprocess-toolkit/preprocess-toolkit-stage1-1-0.zip",
    )
    parser.add_argument("--prep-extract-dir", default="/opt/preprocess-toolkit")
    parser.add_argument("--prep-install-requirements", type=_str_to_bool, default=True)
    parser.add_argument("--prep-timeout-s", type=int, default=0)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--grad-checkpoint", type=_str_to_bool, default=DEFAULTS["grad_checkpoint"])
    parser.add_argument("--dry-run", type=_str_to_bool, default=False)
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--early-stop-ppl", type=float, default=0.0)
    parser.add_argument("--metrics-interval", type=int, default=100)
    return parser.parse_args(argv)


def _parse_betas(betas: str) -> tuple[float, float]:
    parts = [float(x) for x in betas.split(",")]
    if len(parts) != 2:
        raise ValueError("Betas must contain two comma separated values")
    return parts[0], parts[1]


def _str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _log_data_bootstrap(
    toolkit_uri: str,
    extract_dir: str,
    install_requirements: bool,
    mode: str,
    summary: dict,
) -> None:
    logger = configure_logging()
    logger.info("========== DATA BOOTSTRAP ==========")
    logger.info(
        "toolkit=%s | extract_dir=%s | install_requirements=%s",
        toolkit_uri,
        extract_dir,
        install_requirements,
    )
    logger.info("prepare_data=%s", mode)
    datasets = summary.get("datasets", {})
    for job, info in datasets.items():
        status = str(info.get("status", "UNKNOWN")).upper()
        out_dir = info.get("output")
        file_count = info.get("after_files", info.get("before_files", info.get("record_count", 0)))
        logger.info("  %s | status=%s | output=%s | count=%s", job, status, out_dir, file_count)
    logger.info("====================================")


def _augment_data_readiness(
    prep_summary: dict,
    dataset: data.ManifestDataset,
    manifest_snapshot: list[dict],
    toolkit_zip: str,
) -> dict:
    summary = copy.deepcopy(prep_summary)
    summary["toolkit_zip_uri"] = toolkit_zip
    summary["manifest_size"] = len(manifest_snapshot)
    summary["total_records"] = len(dataset)
    summary.setdefault("prepare_data_mode", summary.get("mode"))
    datasets_summary: Dict[str, dict] = summary.setdefault("datasets", {})
    key_to_path: Dict[str, str] = {}
    for entry in dataset.entries:
        key = entry.dataset_id or entry.resolved_path
        key_to_path.setdefault(key, entry.resolved_path)
    for key, count in dataset.dataset_counts.items():
        ds_entry = datasets_summary.setdefault(key, {})
        if "output" not in ds_entry:
            ds_entry["output"] = key_to_path.get(key)
        ds_entry["record_count"] = count
        auto_count = dataset.auto_sample_counts.get(key, 0)
        if auto_count:
            ds_entry["auto_generated_sample_ids"] = auto_count
    return summary


def _write_data_artifacts(
    run_dir: str,
    data_readiness: dict,
    datasets_cfg: Dict[str, DatasetSpec],
    manifest_snapshot: list[dict],
) -> None:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    readiness_path = path / "data_readiness.json"
    readiness_path.write_text(json.dumps(data_readiness, indent=2, sort_keys=True), encoding="utf-8")
    yaml_snapshot = {
        job: {
            "in": spec.inp,
            "out": spec.out,
            "type": spec.dtype,
            "manifest": spec.manifest,
        }
        for job, spec in datasets_cfg.items()
    }
    with (path / "datasets_yaml_snapshot.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(yaml_snapshot, fh, sort_keys=True)
    manifest_path = path / "manifest_snapshot.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for entry in manifest_snapshot:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")


def _detect_git_commit() -> Optional[str]:
    try:
        for candidate in Path(__file__).resolve().parents:
            if (candidate / ".git").exists():
                result = subprocess.run(
                    ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    return result.stdout.strip()
                break
    except Exception:
        return None
    return None


def _write_args_snapshot(run_dir: str, args: argparse.Namespace, run_uri: Optional[str]) -> None:
    snapshot = vars(args).copy()
    env_snapshot = {
        "HF_TOKEN_PRESENT": "true" if bool(os.environ.get("HF_TOKEN")) else "false",
        "GOOGLE_CLOUD_PROJECT": os.environ.get("GOOGLE_CLOUD_PROJECT"),
    }
    commit = _detect_git_commit()
    if commit:
        env_snapshot["git_commit"] = commit
    snapshot["env"] = env_snapshot
    path = Path(run_dir) / "args_snapshot.json"
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    if run_uri:
        local_to_gcs(str(path), f"{run_uri}/args_snapshot.json")


def _log_startup_banner(
    seed: int,
    device_info,
    precision: str,
    sdpa_enabled: bool,
    fa_installed: bool,
    toolkit_zip: str,
    prepare_mode: str,
    data_summary: dict,
    teacher_mode: str,
    teacher_id: str,
    kd_temperature: float,
    alpha_start: float,
    alpha_end: float,
    anneal_pct: float,
    grad_accum_steps: int,
    metrics_interval: int,
    dry_run: bool,
) -> None:
    logger = configure_logging()
    logger.info("========== DATA + KD STARTUP ==========")
    logger.info(
        "seed=%d | device=%s (%s) | precision=%s | sdpa=%s | flash_attn_wheel=%s",
        seed,
        device_info.name,
        device_info.device.type,
        precision,
        "active" if sdpa_enabled else "fallback",
        "installed" if fa_installed else "missing",
    )
    logger.info("grad_accum_steps=%d | metrics_interval=%d | dry_run=%s", grad_accum_steps, metrics_interval, dry_run)
    logger.info("toolkit_zip=%s", toolkit_zip)
    logger.info("prepare_data=%s", prepare_mode)
    datasets = data_summary.get("datasets", {})
    for key, info in sorted(datasets.items()):
        status = str(info.get("status", "ready")).upper()
        out_dir = info.get("output")
        files = info.get("after_files", info.get("before_files", info.get("record_count", 0)))
        auto_ids = info.get("auto_generated_sample_ids")
        extra = f" | auto_sample_ids={auto_ids}" if auto_ids else ""
        logger.info("  dataset=%s | status=%s | output=%s | count=%s%s", key, status, out_dir, files, extra)
    logger.info(
        "teacher_mode=%s | teacher_id=%s | kd_temp=%.2f | alpha_start=%.2f | alpha_end=%.2f | anneal_pct=%.2f",
        teacher_mode,
        teacher_id,
        kd_temperature,
        alpha_start,
        alpha_end,
        anneal_pct,
    )
    logger.info("======================================")


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    logger = configure_logging()
    teacher_mode = args.teacher_mode
    if teacher_mode not in {"online", "precompute"}:
        raise ValueError(f"Invalid teacher_mode: {teacher_mode}")
    if teacher_mode == "precompute" and not args.teacher_logits_dir:
        raise ValueError("teacher_logits_dir is required when teacher_mode=precompute")
    logger.info("Setting random seed to %d", args.seed)
    set_seed(args.seed)
    login_hf()
    fa_installed = install_flash_attn_from_gcs(args.fa_wheel_gcs)
    sdpa_enabled = enable_flash_attn_if_available(log=False)
    device_info = detect_training_device()
    logger.info(
        "Stage-1 KD starting | device=%s | capability=%s | seq_len=%d | teacher_mode=%s",
        device_info.name,
        device_info.capability,
        args.seq_len,
        teacher_mode,
    )
    run_id = os.environ.get("AIP_TRAINING_JOB_ID", "local")
    local_root = ensure_local_dir(os.path.join("/tmp", "vertex_run", run_id))
    os.environ["STAGE1_DATA_PROVENANCE_DIR"] = local_root
    toolkit_zip = normalize_gcs_uri(args.prep_toolkit_zip_uri)
    toolkit_dir = ensure_toolkit(toolkit_zip, args.prep_extract_dir, args.prep_install_requirements)
    datasets_cfg = load_datasets_yaml(toolkit_dir)
    prep_summary = prepare_if_needed(
        args.prepare_data,
        toolkit_dir,
        datasets_cfg,
        timeout_s=args.prep_timeout_s,
    )
    _log_data_bootstrap(toolkit_zip, args.prep_extract_dir, args.prep_install_requirements, args.prepare_data, prep_summary)
    tokenizer = data.build_tokenizer(args.teacher_id)
    manifest_entries, manifest_snapshot = data.read_manifest(args.dataset_manifest, datasets_cfg)
    dataset = data.ManifestDataset(
        manifest_entries,
        tokenizer,
        seq_len=args.seq_len,
        tool_use_ratio=args.tool_use_ratio,
        teacher_mode=args.teacher_mode,
        teacher_logits_dir=args.teacher_logits_dir if args.teacher_mode == "precompute" else None,
    )
    data_readiness = _augment_data_readiness(prep_summary, dataset, manifest_snapshot, toolkit_zip)
    data_readiness["dataset_manifest"] = args.dataset_manifest
    if args.val_manifest:
        data_readiness["val_manifest"] = args.val_manifest
    _write_data_artifacts(local_root, data_readiness, datasets_cfg, manifest_snapshot)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": device_info.device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(dataset, **loader_kwargs)
    val_loader: Optional[DataLoader] = None
    if args.val_manifest:
        val_entries, _ = data.read_manifest(args.val_manifest, datasets_cfg)
        val_dataset = data.ManifestDataset(
            val_entries,
            tokenizer,
            seq_len=args.seq_len,
            tool_use_ratio=0.0,
            teacher_mode="precompute",
        )
        val_kwargs = {
            "batch_size": args.batch_size,
            "shuffle": False,
            "num_workers": min(args.num_workers, 2),
            "pin_memory": device_info.device.type == "cuda",
        }
        if val_kwargs["num_workers"] > 0:
            val_kwargs["prefetch_factor"] = args.prefetch_factor
        val_loader = DataLoader(val_dataset, **val_kwargs)
    elif len(dataset) > 0:
        from torch.utils.data import Subset

        subset_size = min(128, len(dataset))
        val_indices = list(range(subset_size))
        val_kwargs = {
            "batch_size": args.batch_size,
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": device_info.device.type == "cuda",
        }
        val_loader = DataLoader(Subset(dataset, val_indices), **val_kwargs)
    _log_startup_banner(
        args.seed,
        device_info,
        args.precision,
        sdpa_enabled,
        fa_installed,
        toolkit_zip,
        args.prepare_data,
        data_readiness,
        teacher_mode,
        args.teacher_id,
        float(args.kd_temperature),
        float(args.kd_alpha_start),
        float(args.kd_alpha_end),
        float(args.kd_anneal_pct),
        int(args.grad_accum_steps),
        int(args.metrics_interval),
        bool(args.dry_run),
    )
    output_gcs_root = args.output_gcs_uri.rstrip("/")
    run_uri = f"{output_gcs_root}/{run_id}" if output_gcs_root else None
    _write_args_snapshot(local_root, args, run_uri)
    model = initialize_student(
        args.resume_gcs_uri,
        local_root,
        seq_len=args.seq_len,
        output_gcs_uri=output_gcs_root,
        run_id=run_id,
    )
    grad_checkpoint = bool(args.grad_checkpoint)
    model = apply_grad_checkpointing(model, grad_checkpoint)
    betas = _parse_betas(args.betas)
    kd_alpha_schedule = AnnealingSchedule(args.kd_alpha_start, args.kd_alpha_end, args.kd_anneal_pct)
    ce_beta_schedule = AnnealingSchedule(1 - args.kd_alpha_start, 1 - args.kd_alpha_end, args.kd_anneal_pct)
    logit_l2_schedule = AnnealingSchedule(args.keep_old_logit_l2, 0.0, args.keep_old_logit_l2_fade_step / max(1, args.max_steps))
    teacher = None
    if teacher_mode == "online":
        teacher = TeacherWrapper(TeacherConfig(model_id=args.teacher_id))
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device_info.device,
        output_dir=local_root,
        output_gcs_uri=output_gcs_root,
        run_id=run_id,
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
        teacher=teacher,
        teacher_mode=teacher_mode,
        teacher_logits_dir=args.teacher_logits_dir if teacher_mode == "precompute" else None,
        eval_every=args.eval_every,
        save_every=args.save_every,
        grad_accum_steps=int(args.grad_accum_steps),
        metrics_interval=int(args.metrics_interval),
        limit_batches=int(args.limit_batches),
        early_stop_ppl=float(args.early_stop_ppl),
        dry_run=bool(args.dry_run),
    )
    json_log(
        logger,
        {
            "teacher_mode": teacher_mode,
            "dataset_size": len(dataset),
            "seq_len": args.seq_len,
            "output_gcs": f"{output_gcs_root}/{run_id}",
            "grad_checkpoint": grad_checkpoint,
            "grad_accum_steps": int(args.grad_accum_steps),
            "metrics_interval": int(args.metrics_interval),
            "dry_run": bool(args.dry_run),
        },
    )
    trainer.train()


if __name__ == "__main__":
    main()
