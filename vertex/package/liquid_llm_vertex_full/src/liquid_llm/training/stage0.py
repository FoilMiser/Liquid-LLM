import os
import random
from pathlib import Path
from datetime import datetime

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

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------
    train_dl, val_dl, vocab_size, pad_id, tok = build_dataloaders(
        teacher_name=teacher_name,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        block_size=block_size,
        global_batch=global_batch,
        seed=seed,
    )

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
    scheduler = build_scheduler(
        optimizer, warmup_steps=warmup_steps, total_steps=train_steps
    )

    # ---------------------------------------------------------------------
    # Resume (optional)
    # ---------------------------------------------------------------------
    step0 = 0
    if resume_gcs_uri:
        try:
            step0 = load_from_uri(student, optimizer, scheduler, resume_gcs_uri)
            log.info(f"Resumed from {resume_gcs_uri} at step {step0}")
        except Exception as e:
            log.warning(f"Resume failed: {e}")

    # ---------------------------------------------------------------------
    # Outputs
    # ---------------------------------------------------------------------
    local_outdir = Path(local_workdir) / "outputs"
    local_outdir.mkdir(parents=True, exist_ok=True)
    gcs_outdir = _expand_output_uri(output_gcs_uri)
    if gcs_outdir:
        log.info(f"Writing checkpoints to {gcs_outdir}")

    # ---------------------------------------------------------------------
    # Training state
    #   NOTE: train_loop expects `state['log']` to be a logger with .info()
    #         Keep custom bookkeeping in `log_state`.
    # ---------------------------------------------------------------------
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
        output_prefix="",
        ckptio=save_and_maybe_upload,
        local_outdir=str(local_outdir),
        gcs_outdir=gcs_outdir,
        precision=precision,
        step=step0,
        micro_batch=micro_batch,
        global_batch=global_batch,
        # logger used by train_loop
        log=get_logger("train"),
        # optional bookkeeping for your own use
        log_state={
            "history": [],
            "last_eval": None,
            "running_loss": None,
        },
    )

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
    save_and_maybe_upload(sd, local_outdir, gcs_outdir, filename="final.pt")
    log.info(f"Finished at step {final_step}.")
