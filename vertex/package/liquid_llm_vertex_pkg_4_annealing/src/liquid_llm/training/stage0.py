import os
import random
import subprocess
from pathlib import Path
from datetime import datetime
from importlib import metadata

import numpy as np
import torch

from transformers import AutoTokenizer  # (kept if other modules rely on it)

from ..data.wikitext import build_dataloaders
from ..models.liquid import build_student_model
from ..training.optim import build_optimizer
from ..training.schedules import build_scheduler
from ..training.loop import train_loop
from ..utils.logging import get_logger
from ..io.checkpoints import load_from_uri, save_and_maybe_upload


def _expand_output_uri(output_gcs_uri: str | None) -> str | None:
    """
    Vertex passes job args directly to Python, so any shell substitutions
    like $(date ...) won't expand. Handle common patterns here, and also
    create a sensible default path if None is provided.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    if output_gcs_uri:
        # Replace a common shell placeholder if present
        if "$(" in output_gcs_uri:
            output_gcs_uri = output_gcs_uri.replace("$(date +%Y%m%d-%H%M%S)", ts)
        return output_gcs_uri

    # If not provided, default to a timestamped run folder under the bucket.
    # Adjust the base prefix to your preference.
    return f"gs://liquid-llm-bucket/liquid-llm/stage0/checkpoints/vertex_runs/{ts}"


def run_training(
    resume_gcs_uri: str | None,
    block_size: int,
    teacher_name: str,
    dataset_name: str,
    dataset_config: str,
    output_gcs_uri: str | None = None,
    local_workdir: str = "/tmp/liquid_work",
    seed: int = 42,
    global_batch: int = 64,
    micro_batch: int = 8,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas=(0.9, 0.95),
    eps: float = 1e-8,
    warmup_steps: int = 2000,
    train_steps: int = 45000,
    eval_every: int = 500,
    save_every: int = 1000,
    log_interval: int = 50,
    precision: str = "fp16",
    model: dict = None,
    hf_token: str | None = None,
    hf_secret_name: str | None = None,
    alpha: float | None = None,
    T: float | None = None,
    alpha_schedule: list | None = None,
    temp_schedule: list | None = None,
    alpha_schedule_raw: str | None = None,
    temp_schedule_raw: str | None = None,
    eval_ctx_lens: list[int] | None = None,
    reset_optim_on_ctx_change: bool = False,
    grad_accum: int | None = None,
    lr_peak: float | None = None,
    rope_scale: float | None = None,
):
    log = get_logger("stage0")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------------------
    # Seeding
    # ---------------------------------------------------------------------
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def _normalize_schedule(items):
        normalized: list[tuple[int, float]] = []
        if not items:
            return normalized
        for entry in items:
            if isinstance(entry, dict):
                step, value = next(iter(entry.items()))
            else:
                step, value = entry
            normalized.append((int(step), float(value)))
        normalized.sort(key=lambda x: x[0])
        return normalized

    def _schedule_to_string(items):
        if not items:
            return None
        return ",".join(f"{int(step)}:{float(value)}" for step, value in items)

    def _consume_schedule(items, completed_steps: int, current_value: float):
        remaining: list[tuple[int, float]] = []
        updated = current_value
        for trigger, value in items:
            if completed_steps >= trigger:
                updated = float(value)
            else:
                remaining.append((int(trigger), float(value)))
        return updated, remaining

    alpha_schedule = _normalize_schedule(alpha_schedule)
    temp_schedule = _normalize_schedule(temp_schedule)
    eval_ctx_lens = sorted({int(x) for x in (eval_ctx_lens or [])})

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------
    if hf_token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
        os.environ.setdefault("HF_TOKEN", hf_token)

    def _build_primary_loaders(target_block: int):
        return build_dataloaders(
            teacher_name=teacher_name,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            block_size=target_block,
            global_batch=global_batch,
            seed=seed,
            hf_token=hf_token,
        )

    (
        train_dl,
        val_dl,
        vocab_size,
        pad_id,
        tok,
        val_split,
    ) = _build_primary_loaders(block_size)

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------
    model = model or {}
    margs = dict(
        d_model=model.get("d_model", 768),
        n_layers=model.get("n_layers", 10),
        n_heads=model.get("n_heads", 12),
        dropout=model.get("dropout", 0.0),
    )
    student = build_student_model(
        vocab_size=vocab_size,
        pad_id=pad_id,
        **margs,
    ).to(device)

    # ---------------------------------------------------------------------
    # Optimizer / Scheduler
    # ---------------------------------------------------------------------
    optimizer = build_optimizer(
        student, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
    )
    lr_peak_value = lr_peak
    peak_scale = 1.0
    if lr_peak_value is not None and lr > 0:
        peak_scale = float(lr_peak_value) / float(lr)
    scheduler = build_scheduler(
        optimizer, warmup_steps=warmup_steps, total_steps=train_steps, peak_scale=peak_scale
    )

    # ---------------------------------------------------------------------
    # Resume (optional)
    # ---------------------------------------------------------------------
    step0 = 0
    checkpoint_meta: dict = {}
    trainer_resume_state: dict = {}
    if resume_gcs_uri:
        try:
            step0, resume_payload = load_from_uri(student, optimizer, scheduler, resume_gcs_uri)
            trainer_resume_state = resume_payload.get('trainer_state', {}) or {}
            checkpoint_meta = resume_payload.get('meta', {}) or {}
            log.info("[resume] Loaded checkpoint from %s at step %s", resume_gcs_uri, step0)
            if checkpoint_meta:
                log.info("[resume-meta] %s", checkpoint_meta)
        except Exception as e:
            log.warning(f"Resume failed: {e}")
            checkpoint_meta = {}
            trainer_resume_state = {}

    kd_alpha_default = trainer_resume_state.get("kd_alpha")
    if kd_alpha_default is None:
        kd_alpha_default = checkpoint_meta.get("kd_alpha", 0.5)
    kd_temperature_default = trainer_resume_state.get("kd_temperature")
    if kd_temperature_default is None:
        kd_temperature_default = checkpoint_meta.get("kd_temperature", 1.0)

    kd_alpha_value = alpha if alpha is not None else kd_alpha_default
    if kd_alpha_value is None:
        kd_alpha_value = 0.5
    kd_temperature_value = T if T is not None else kd_temperature_default
    if kd_temperature_value is None:
        kd_temperature_value = 1.0

    if not alpha_schedule:
        alpha_schedule = _normalize_schedule(trainer_resume_state.get("kd_alpha_schedule"))
    if not temp_schedule:
        temp_schedule = _normalize_schedule(trainer_resume_state.get("kd_temperature_schedule"))

    alpha_schedule_original = alpha_schedule_raw if isinstance(alpha_schedule_raw, str) else None
    if alpha_schedule_original is None:
        alpha_schedule_original = checkpoint_meta.get("alpha_schedule")
    if alpha_schedule_original is None:
        alpha_schedule_original = _schedule_to_string(alpha_schedule)

    temp_schedule_original = temp_schedule_raw if isinstance(temp_schedule_raw, str) else None
    if temp_schedule_original is None:
        temp_schedule_original = checkpoint_meta.get("temp_schedule")
    if temp_schedule_original is None:
        temp_schedule_original = _schedule_to_string(temp_schedule)

    grad_accum_value = grad_accum if grad_accum is not None else trainer_resume_state.get(
        "grad_accum", checkpoint_meta.get("grad_accum", 1)
    )
    grad_accum_value = max(1, int(grad_accum_value))

    if lr_peak_value is None:
        lr_peak_value = checkpoint_meta.get("lr_peak")

    if rope_scale is None:
        rope_scale = checkpoint_meta.get("rope_scale")

    ckpt_block_size = checkpoint_meta.get("block_size")
    if ckpt_block_size and ckpt_block_size != block_size:
        log.info(
            "[ctx] Requested block_size=%s differs from checkpoint block_size=%s; rebuilding data loaders.",
            block_size,
            ckpt_block_size,
        )
        (
            train_dl,
            val_dl,
            vocab_size,
            pad_id,
            tok,
            val_split,
        ) = _build_primary_loaders(block_size)
        resized = False
        if hasattr(student, "resize_positional_embeddings"):
            try:
                resized = student.resize_positional_embeddings(block_size)
            except Exception as resize_exc:  # pragma: no cover - defensive logging
                log.warning("Failed to resize positional embeddings: %s", resize_exc)
        if resized:
            log.info("[ctx] Resized positional embeddings to support block_size=%s", block_size)
        if reset_optim_on_ctx_change:
            optimizer = build_optimizer(
                student, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
            )
            peak_scale = 1.0
            if lr_peak_value is not None and lr > 0:
                peak_scale = float(lr_peak_value) / float(lr)
            scheduler = build_scheduler(
                optimizer, warmup_steps=warmup_steps, total_steps=train_steps, peak_scale=peak_scale
            )
            step0 = 0
            trainer_resume_state = {}
            log.info("[ctx] Optimizer and scheduler reset after context-length change; training step reset to 0")

    if hasattr(student, "resize_positional_embeddings"):
        current_max = getattr(student.pos, "num_embeddings", block_size)
        if block_size > current_max:
            if student.resize_positional_embeddings(block_size):
                log.info("[ctx] Expanded positional embeddings to %s tokens", block_size)

    if step0 > 0:
        if alpha is None:
            kd_alpha_value, alpha_schedule = _consume_schedule(
                alpha_schedule, step0, float(kd_alpha_value)
            )
        if T is None:
            kd_temperature_value, temp_schedule = _consume_schedule(
                temp_schedule, step0, float(kd_temperature_value)
            )

    eval_ctx_loaders: dict[int, object] = {}
    for ctx_len in eval_ctx_lens:
        if ctx_len <= 0:
            continue
        if ctx_len == block_size:
            eval_ctx_loaders[ctx_len] = val_dl
        else:
            _, eval_loader, _, _, _, _ = _build_primary_loaders(ctx_len)
            eval_ctx_loaders[ctx_len] = eval_loader
            log.info("[eval] Prepared validation loader for context length %s", ctx_len)

    pos_embedding_meta = {
        "type": "absolute",
        "num_embeddings": int(getattr(student.pos, "num_embeddings", block_size)),
        "embedding_dim": int(getattr(student.pos, "embedding_dim", margs.get("d_model", 0))),
        "buffer_positions": int(student.pos_ids.size(-1)) if hasattr(student, "pos_ids") else None,
    }

    log.info(
        "[anneal] kd_alpha=%s kd_temperature=%s grad_accum=%s rope_scale=%s",
        kd_alpha_value,
        kd_temperature_value,
        grad_accum_value,
        rope_scale if rope_scale is not None else "na",
    )

    # ---------------------------------------------------------------------
    # Outputs
    # ---------------------------------------------------------------------
    local_outdir = Path(local_workdir) / "outputs"
    local_outdir.mkdir(parents=True, exist_ok=True)
    gcs_outdir = _expand_output_uri(output_gcs_uri)

    # ---------------------------------------------------------------------
    # Training state
    #   NOTE: train_loop expects `state['log']` to be a logger with .info()
    #         Keep custom bookkeeping in `log_state`.
    # ---------------------------------------------------------------------
    try:
        pkg_ver = metadata.version("liquid_llm_vertex_pkg_4_annealing")
    except metadata.PackageNotFoundError:
        pkg_ver = "unknown"

    repo_root = None
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            repo_root = candidate
            break
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[4]

    try:
        git_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
            .decode()
            .strip()
        )
    except Exception:
        git_sha = "unknown"

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{random.getrandbits(16):04x}"

    state = dict(
        model=student,
        device=device,
        teacher_name=teacher_name,
        train_loader=train_dl,
        val_loader=val_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        save_every=save_every,
        eval_every=eval_every,
        log_interval=log_interval,
        train_steps=train_steps,
        block_size=block_size,
        output_prefix="",
        ckptio=save_and_maybe_upload,
        local_outdir=str(local_outdir),
        gcs_outdir=gcs_outdir,
        precision=precision,
        step=step0,
        micro_batch=micro_batch,
        global_batch=global_batch,
        hf_token=hf_token,
        pad_id=pad_id,
        dataset_name=dataset_name,
        val_split=val_split,
        grad_clip_norm=1.0,
        seed=seed,
        run_id=run_id,
        kd_alpha=float(kd_alpha_value),
        kd_temperature=float(kd_temperature_value),
        kd_alpha_schedule=list(alpha_schedule),
        kd_temperature_schedule=list(temp_schedule),
        kd_alpha_schedule_original=alpha_schedule_original,
        kd_temperature_schedule_original=temp_schedule_original,
        eval_ctx_loaders=eval_ctx_loaders,
        eval_ctx_lens=eval_ctx_lens,
        grad_accum=grad_accum_value,
        lr=lr,
        lr_peak=lr_peak_value,
        warmup_steps=warmup_steps,
        pos_embedding_meta=pos_embedding_meta,
        resume_meta=checkpoint_meta,
        kd_resume_state=trainer_resume_state,
        rope_scale=rope_scale,
        run_meta={
            "git_sha": git_sha,
            "pkg_ver": pkg_ver,
            "resume_uri": resume_gcs_uri,
        },
        # logger used by train_loop
        log=get_logger("train"),
        # optional bookkeeping for your own use
        log_state={
            "history": [],
            "last_eval": None,
            "running_loss": None,
        },
    )

    def _build_checkpoint_payload(step_value: int):
        remaining_alpha = [(int(s), float(v)) for s, v in state.get("kd_alpha_schedule", [])]
        remaining_temp = [(int(s), float(v)) for s, v in state.get("kd_temperature_schedule", [])]
        trainer_state = {
            "kd_alpha": float(state.get("kd_alpha", 0.5)),
            "kd_temperature": float(state.get("kd_temperature", 1.0)),
            "kd_alpha_schedule": remaining_alpha,
            "kd_temperature_schedule": remaining_temp,
            "grad_accum": int(state.get("grad_accum", 1)),
        }
        meta = {
            "block_size": state.get("block_size"),
            "kd_alpha": trainer_state["kd_alpha"],
            "kd_temperature": trainer_state["kd_temperature"],
            "alpha_schedule": state.get("kd_alpha_schedule_original"),
            "temp_schedule": state.get("kd_temperature_schedule_original"),
            "grad_accum": trainer_state["grad_accum"],
            "pos_embedding": state.get("pos_embedding_meta"),
            "lr": state.get("lr"),
            "lr_peak": state.get("lr_peak"),
            "warmup_steps": state.get("warmup_steps"),
            "train_steps": state.get("train_steps"),
            "eval_ctx_lens": state.get("eval_ctx_lens"),
        }
        rope_scale = state.get("rope_scale")
        if rope_scale is not None:
            meta["rope_scale"] = rope_scale
        meta = {k: v for k, v in meta.items() if v is not None}
        return {"trainer_state": trainer_state, "meta": meta}

    state["checkpoint_extra_builder"] = _build_checkpoint_payload

    # ---------------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------------
    final_step = train_loop(state)

    # ---------------------------------------------------------------------
    # Final save
    # ---------------------------------------------------------------------
    sd = {
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": final_step,
        "tokenizer": getattr(tok, "name_or_path", None),
    }
    meta_payload = {}
    builder = state.get("checkpoint_extra_builder")
    if builder is not None:
        meta_payload = builder(final_step)
    trainer_state_ckpt = meta_payload.get("trainer_state") if meta_payload else None
    meta_ckpt = meta_payload.get("meta") if meta_payload else None
    if trainer_state_ckpt:
        sd["trainer_state"] = trainer_state_ckpt
    if meta_ckpt:
        sd["meta"] = meta_ckpt
    final_local, final_uri = save_and_maybe_upload(
        sd, local_outdir, gcs_outdir, filename="final.pt", meta=meta_ckpt
    )
    if meta_ckpt:
        log.info("[ckpt-meta] step=%s filename=final.pt meta=%s", final_step, meta_ckpt)
    if gcs_outdir and not final_uri:
        log.warning(
            "[ckpt] expected to upload final checkpoint to %s but no URI was returned",
            gcs_outdir,
        )
    state.setdefault("log_state", {}).update(
        {
            "final_checkpoint": {
                "step": final_step,
                "local_path": final_local,
                "gcs_uri": final_uri,
            }
        }
    )
    log.info(f"Finished at step {final_step}.")
