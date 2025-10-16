"""Precompute KD logits for math/code splits using the hybrid teacher."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from . import data
from .gcs_io import GCSIOError, ensure_local_dir, list_gcs, local_to_gcs
from .prep import ensure_toolkit, load_datasets_yaml, normalize_gcs_uri, prepare_if_needed
from .teacher import TeacherConfig, TeacherWrapper
from .utils import configure_logging

logger = configure_logging()

DEFAULT_TOOLKIT_ZIP = "gs://liquid-llm-bucket-2/sandbox/preprocess-toolkit/preprocess-toolkit-stage1-1-0.zip"
TMP_DIR = ensure_local_dir("/tmp/hybrid_logits")


def _str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _logit_exists(base_dir: str, sample_id: str) -> bool:
    base = base_dir.rstrip("/")
    for ext in (".pt", ".npy"):
        candidate = f"{base}/{sample_id}{ext}"
        if candidate.startswith("gs://"):
            try:
                matches = list_gcs(candidate)
            except GCSIOError:
                matches = []
            if matches:
                return True
        else:
            if Path(candidate).exists():
                return True
    return False


def _save_logits(tensor: torch.Tensor, base_dir: str, sample_id: str) -> None:
    dest = f"{base_dir.rstrip('/')}/{sample_id}.pt"
    payload = tensor.detach().cpu()
    if dest.startswith("gs://"):
        tmp_path = Path(TMP_DIR) / f"{sample_id}.pt"
        torch.save(payload, tmp_path)
        try:
            local_to_gcs(str(tmp_path), dest)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    else:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, dest)


def _filter_entries(manifest_entries: Sequence[data.ManifestEntry]) -> List[data.ManifestEntry]:
    allowed = {"math_tool", "code"}
    filtered = [entry for entry in manifest_entries if entry.type in allowed]
    if not filtered:
        logger.warning("No math/code entries found in manifest; nothing to precompute")
    else:
        logger.info("Filtered %d entries for hybrid precompute", len(filtered))
    return filtered


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute hybrid KD logits for math/code datasets")
    parser.add_argument("--mc-teacher-id", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--math-logits-dir", required=True)
    parser.add_argument("--code-logits-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--overwrite", type=_str_to_bool, default=False)
    parser.add_argument("--prep-toolkit-zip-uri", default=DEFAULT_TOOLKIT_ZIP)
    parser.add_argument("--prep-extract-dir", default="/opt/preprocess-toolkit")
    parser.add_argument("--prep-install-requirements", type=_str_to_bool, default=True)
    parser.add_argument("--prep-timeout-s", type=int, default=0)
    parser.add_argument("--limit-samples", type=int, default=0)
    return parser.parse_args(argv)


def _build_dataloader(
    manifest_entries: Sequence[data.ManifestEntry],
    tokenizer,
    seq_len: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    limit_samples: int,
) -> DataLoader:
    dataset = data.ManifestDataset(
        manifest_entries,
        tokenizer,
        seq_len=seq_len,
        tool_use_ratio=0.0,
        teacher_mode="online",
        hybrid=False,
        split="precompute",
    )
    if limit_samples:
        from torch.utils.data import Subset

        limit = min(limit_samples, len(dataset))
        dataset = Subset(dataset, list(range(limit)))
    loader_kwargs: Dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": data.collate_batch,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def _pending_indices(
    sample_ids: Sequence[str],
    sample_types: Sequence[str],
    math_dir: str,
    code_dir: str,
    overwrite: bool,
) -> Tuple[List[int], List[Tuple[str, str]]]:
    pending: List[int] = []
    outputs: List[Tuple[str, str]] = []
    for idx, (sid, stype) in enumerate(zip(sample_ids, sample_types)):
        if stype == "math_tool":
            base_dir = math_dir
        elif stype == "code":
            base_dir = code_dir
        else:
            logger.debug("Skipping sample %s with unsupported type %s", sid, stype)
            continue
        if not overwrite and _logit_exists(base_dir, sid):
            continue
        pending.append(idx)
        outputs.append((base_dir, sid))
    return pending, outputs


def run(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logger.info(
        "Precomputing hybrid logits | teacher=%s | manifest=%s | math_dir=%s | code_dir=%s",
        args.mc_teacher_id,
        args.dataset_manifest,
        args.math_logits_dir,
        args.code_logits_dir,
    )
    toolkit_zip = normalize_gcs_uri(args.prep_toolkit_zip_uri)
    toolkit_dir = ensure_toolkit(toolkit_zip, args.prep_extract_dir, args.prep_install_requirements)
    datasets_cfg = load_datasets_yaml(toolkit_dir)
    prepare_if_needed(
        "skip",
        toolkit_dir,
        datasets_cfg,
        timeout_s=args.prep_timeout_s,
    )
    manifest_entries, _ = data.read_manifest(args.dataset_manifest, datasets_cfg)
    filtered_entries = _filter_entries(manifest_entries)
    if not filtered_entries:
        return
    tokenizer = data.build_tokenizer(args.mc_teacher_id)
    dataloader = _build_dataloader(
        filtered_entries,
        tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        limit_samples=args.limit_samples,
    )
    teacher = TeacherWrapper(TeacherConfig(model_id=args.mc_teacher_id))
    processed = 0
    skipped = 0
    start = time.perf_counter()
    for step, batch in enumerate(dataloader, start=1):
        sample_ids = batch["sample_ids"]
        sample_types = batch.get("sample_types", [])
        pending, outputs = _pending_indices(
            sample_ids,
            sample_types,
            args.math_logits_dir,
            args.code_logits_dir,
            bool(args.overwrite),
        )
        skipped += len(sample_ids) - len(pending)
        if not pending:
            continue
        with torch.inference_mode():
            logits = teacher.logits(batch["input_ids"][pending])
        for tensor, (base_dir, sample_id) in zip(logits, outputs):
            _save_logits(tensor, base_dir, sample_id)
        processed += len(outputs)
        if step % 50 == 0:
            logger.info(
                "Processed %d samples (skipped=%d) | last_batch=%d", processed, skipped, len(outputs)
            )
    elapsed = max(time.perf_counter() - start, 1e-6)
    if processed:
        logger.info(
            "Completed hybrid precompute: saved=%d | skipped=%d | rate=%.2f samples/s",
            processed,
            skipped,
            processed / elapsed,
        )
    else:
        logger.warning("No logits were generated; check manifest and output directories")


if __name__ == "__main__":
    run()
