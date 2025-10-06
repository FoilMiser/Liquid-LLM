import time
from pathlib import Path

import torch

from .evaluation import evaluate
from .metrics import Meter, count_active_tokens, cross_entropy, get_logits


def _save_ckpt(ckptio, state, step, filename, log):
    """Helper to save a checkpoint and log locations."""
    state_dict = {
        "model": state["model"].state_dict(),
        "optimizer": state["optimizer"].state_dict(),
        "scheduler": state["scheduler"].state_dict(),
        "step": step,
    }
    local, uri = ckptio(state_dict, state["local_outdir"], state["gcs_outdir"], filename=filename)
    log.info(f"[ckpt] saved {local}" + (f" and uploaded to {uri}" if uri else ""))
    return local


# -----------------------------
# Unified pruning helpers
# -----------------------------
def _list_sorted_by_mtime(pattern: str, local_outdir: str):
    """Return Path list sorted by mtime (oldest first)."""
    return sorted(Path(local_outdir).glob(pattern), key=lambda p: p.stat().st_mtime)


def _prune_files(files, keep_k: int | None, retention_secs: int | None, now: float, log, label: str):
    """
    Apply (1) optional age-based prune, then (2) optional count-based prune.
    Files should be oldest-first.
    """
    removed = 0
    survivors = list(files)

    # 1) Age-based prune
    if retention_secs is not None and retention_secs > 0:
        tmp = []
        for p in survivors:
            try:
                age = now - p.stat().st_mtime
                if age > retention_secs:
                    p.unlink(missing_ok=True)
                    removed += 1
                else:
                    tmp.append(p)
            except Exception as e:
                log.warning(f"[ckpt] {label} prune skip {p}: {e}")
                tmp.append(p)
        survivors = tmp

    # 2) Count-based prune (keep most recent K by mtime)
    if keep_k is not None and keep_k >= 0:
        if keep_k == 0:
            for p in survivors:
                try:
                    p.unlink(missing_ok=True)
                    removed += 1
                except Exception as e:
                    log.warning(f"[ckpt] {label} prune skip {p}: {e}")
            survivors = []
        elif len(survivors) > keep_k:
            to_delete = survivors[: len(survivors) - keep_k]  # oldest first
            for p in to_delete:
                try:
                    p.unlink(missing_ok=True)
                    removed += 1
                except Exception as e:
                    log.warning(f"[ckpt] {label} prune skip {p}: {e}")
            survivors = survivors[len(survivors) - keep_k :]

    if removed:
        log.info(f"[ckpt] pruned {removed} old {label} checkpoints.")
    return survivors


def _prune_time_ckpts(local_outdir: str, log, retention_secs: int | None, keep_k: int | None, now: float | None = None):
    """
    Prune *time-based* checkpoints matching `ckpt_time_*.pt`.
    """
    now = now or time.time()
    files = _list_sorted_by_mtime("ckpt_time_*.pt", local_outdir)
    _prune_files(files, keep_k=keep_k, retention_secs=retention_secs, now=now, log=log, label="time-based")


def _prune_best_ckpts(local_outdir: str, log, retention_secs: int | None, keep_k: int | None, now: float | None = None):
    """
    Prune versioned best checkpoints saved as `ckpt_best_*.pt`.
    NOTE: Does NOT remove the rolling `best.pt`.
    """
    now = now or time.time()
    files = _list_sorted_by_mtime("ckpt_best_*.pt", local_outdir)
    if not files:
        return
    _prune_files(files, keep_k=keep_k, retention_secs=retention_secs, now=now, log=log, label="best")


def _prune_step_ckpts(local_outdir: str, log, retention_secs: int | None, keep_k: int | None, now: float | None = None):
    """
    Prune step-based checkpoints saved as `ckpt_step_*.pt`.
    """
    now = now or time.time()
    files = _list_sorted_by_mtime("ckpt_step_*.pt", local_outdir)
    if not files:
        return
    _prune_files(files, keep_k=keep_k, retention_secs=retention_secs, now=now, log=log, label="step-based")


def train_loop(state):
    model         = state["model"]
    device        = state["device"]
    teacher       = state.get("teacher")
    kd_alpha      = float(state.get("kd_alpha", 0.5))
    kd_temperature = float(state.get("kd_temperature", 1.0))
    train_loader  = state["train_loader"]
    val_loader    = state["val_loader"]
    optimizer     = state["optimizer"]
    scheduler     = state["scheduler"]
    save_every    = state["save_every"]
    eval_every    = state["eval_every"]
    log           = state["log"]
    total_steps   = state["train_steps"]
    ckptio        = state["ckptio"]
    precision     = state.get("precision", "fp16")
    grad_clip     = float(state.get("grad_clip", 1.0))
    teacher_eval_every = int(state.get("teacher_eval_every", 0) or 0)
    teacher_metrics = state.get("teacher_metrics")
    precision = precision or "no"

    device_type = "cuda" if device.startswith("cuda") else device
    amp_dtype = None
    amp_enabled = device_type == "cuda" and precision in {"fp16", "bf16"}
    if amp_enabled:
        amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    # --- Time-based checkpoint cadence & retention ---
    raw_time_ckpt_secs = state.get("time_ckpt_secs", 1800)
    time_ckpt_secs = int(raw_time_ckpt_secs) if raw_time_ckpt_secs else 0
    time_ckpt_retention_secs  = state.get("time_ckpt_retention_secs", 14400)      # 4 hours by default
    time_ckpt_keep_k          = state.get("time_ckpt_keep_k", None)               # default: count-unbounded
    if time_ckpt_retention_secs is not None:
        time_ckpt_retention_secs = int(time_ckpt_retention_secs)
    last_time_ckpt_ts         = float(state.get("last_time_ckpt_ts", 0.0))

    # --- Best checkpoint retention controls (versioned snapshots) ---
    raw_best_keep = state.get("best_ckpt_keep_k", 3)
    best_ckpt_keep_k          = int(raw_best_keep) if raw_best_keep is not None else None
    best_ckpt_retention_secs  = state.get("best_ckpt_retention_secs", None)       # optional age prune
    if best_ckpt_retention_secs is not None:
        best_ckpt_retention_secs = int(best_ckpt_retention_secs)

    # --- Step checkpoint retention controls ---
    raw_step_keep = state.get("step_ckpt_keep_k", 5)
    step_ckpt_keep_k          = int(raw_step_keep) if raw_step_keep is not None else None
    step_ckpt_retention_secs  = state.get("step_ckpt_retention_secs", None)       # optional age prune
    if step_ckpt_retention_secs is not None:
        step_ckpt_retention_secs = int(step_ckpt_retention_secs)

    # Track best validation loss for "best.pt"
    best_val_loss = float(state.get("best_val_loss", float("inf")))
    best_path_local = None

    step = state.get("step", 0)
    scaler = torch.amp.GradScaler(device_type, enabled=(precision == "fp16" and device_type == "cuda"))

    # Meters for logging
    tok_meter   = Meter()  # token-weighted mean (comparable to eval CE)
    batch_meter = Meter()  # batch-weighted mean (debug)

    since_log_tokens   = 0
    since_log_examples = 0
    t0 = time.time()
    start_time = t0  # for first time-based ckpt window

    model.train()
    while step < total_steps:
        for batch in train_loader:
            step += 1
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels    = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type, enabled=amp_enabled, dtype=amp_dtype):
                logits_s = get_logits(model(input_ids))
                loss_ce  = cross_entropy(logits_s, labels)

                loss_kd = None
                if teacher is not None and kd_alpha > 0:
                    T = kd_temperature
                    with torch.no_grad():
                        logits_t = get_logits(teacher(input_ids))
                    logprob_s = torch.nn.functional.log_softmax(logits_s / T, dim=-1)
                    prob_t    = torch.nn.functional.softmax(logits_t / T, dim=-1)
                    loss_kd   = torch.nn.functional.kl_div(
                        logprob_s, prob_t, reduction="batchmean"
                    ) * (T * T)
                    loss = (1 - kd_alpha) * loss_ce + kd_alpha * loss_kd
                else:
                    loss = loss_ce

            scaler.scale(loss).backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # ===== Logging & meters =====
            tokens_in_batch = input_ids.numel()
            contrib_tokens  = count_active_tokens(labels, tokens_in_batch)

            tok_meter.update(loss.item(), k=contrib_tokens)
            batch_meter.update(loss.item(), k=input_ids.size(0))

            since_log_tokens   += contrib_tokens
            since_log_examples += input_ids.size(0)

            # Periodic training log
            if step % state["log_interval"] == 0:
                elapsed = max(time.time() - t0, 1e-9)
                tok_per_sec  = since_log_tokens / elapsed
                ex_per_sec   = since_log_examples / elapsed
                msg = (
                    f"step={step}/{total_steps} "
                    f"loss_tok_mean={tok_meter.avg:.4f} "
                    f"loss_batch_mean={batch_meter.avg:.4f} "
                    f"tok/s={tok_per_sec:.0f} "
                    f"ex/s={ex_per_sec:.1f}"
                )
                if loss_kd is not None:
                    msg += f" loss_ce={loss_ce.item():.4f} loss_kd={loss_kd.item():.4f}"
                else:
                    msg += f" loss_ce={loss_ce.item():.4f}"
                log.info(msg)
                tok_meter.reset()
                batch_meter.reset()
                since_log_tokens   = 0
                since_log_examples = 0
                t0 = time.time()

            # Scheduled eval + best checkpointing
            if step % eval_every == 0:
                model.eval()
                metrics = evaluate(model, val_loader, device=device, precision=precision)
                val_loss = float(metrics["val_loss"])
                log.info(
                    f"[eval/student] step={step} val_loss={val_loss:.4f} "
                    f"ppl={metrics['val_ppl']:.2f} tokens={metrics['val_tokens']}"
                )

                # Optionally evaluate the teacher for comparison
                if teacher is not None:
                    recompute_teacher = False
                    if teacher_eval_every > 0 and step % teacher_eval_every == 0:
                        recompute_teacher = True
                    if teacher_metrics is None or recompute_teacher:
                        teacher_metrics = evaluate(
                            teacher,
                            val_loader,
                            device=device,
                            precision=precision,
                        )
                        state["teacher_metrics"] = teacher_metrics

                    if teacher_metrics is not None:
                        delta = val_loss - float(teacher_metrics["val_loss"])
                        log.info(
                            f"[eval/teacher] step={step} val_loss={teacher_metrics['val_loss']:.4f} "
                            f"ppl={teacher_metrics['val_ppl']:.2f} delta_student={delta:+.4f}"
                        )

                # New best? Save/overwrite best.pt and versioned best snapshot
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                    # Rolling "best.pt" (always the current best)
                    best_path_local = _save_ckpt(
                        ckptio, state, step, filename="best.pt", log=log
                    )

                    # Versioned best snapshot for historical/top-K retention
                    vers_name = f"ckpt_best_step{step}_vl{val_loss:.4f}.pt"
                    _save_ckpt(ckptio, state, step, filename=vers_name, log=log)

                    # Prune older best checkpoints per policy
                    try:
                        _prune_best_ckpts(
                            local_outdir=state["local_outdir"],
                            log=log,
                            keep_k=best_ckpt_keep_k,
                            retention_secs=best_ckpt_retention_secs,
                            now=time.time(),
                        )
                    except Exception as e:
                        log.warning(f"[ckpt] best prune error: {e}")

                    log.info(f"[ckpt] new best: val_loss={best_val_loss:.4f} at step={step}")

                model.train()

            # Step-based checkpointing
            if save_every > 0 and step % save_every == 0:
                _save_ckpt(ckptio, state, step, filename=f"ckpt_step_{step}.pt", log=log)
                # Prune older step checkpoints per policy (age &/or count)
                try:
                    _prune_step_ckpts(
                        local_outdir=state["local_outdir"],
                        log=log,
                        retention_secs=step_ckpt_retention_secs,
                        keep_k=step_ckpt_keep_k,
                        now=time.time(),
                    )
                except Exception as e:
                    log.warning(f"[ckpt] step prune error: {e}")

            # Time-based checkpointing (e.g., every 30–60 minutes)
            now_ts = time.time()
            if last_time_ckpt_ts == 0.0:
                last_time_ckpt_ts = start_time
            if time_ckpt_secs > 0 and (now_ts - last_time_ckpt_ts) >= time_ckpt_secs:
                ts = int(now_ts)
                fname = f"ckpt_time_{ts}.pt"
                _save_ckpt(ckptio, state, step, filename=fname, log=log)
                last_time_ckpt_ts = now_ts
                state["last_time_ckpt_ts"] = last_time_ckpt_ts  # persist within this run

                # Prune older time-based checkpoints to save disk
                try:
                    _prune_time_ckpts(
                        local_outdir=state["local_outdir"],
                        log=log,
                        retention_secs=time_ckpt_retention_secs,
                        keep_k=time_ckpt_keep_k,
                        now=now_ts,
                    )
                except Exception as e:
                    log.warning(f"[ckpt] time prune error: {e}")

            if step >= total_steps:
                break

    # Persist best loss to state (helpful if caller saves run metadata)
    state["best_val_loss"] = best_val_loss
    state["teacher_metrics"] = teacher_metrics
    return step
