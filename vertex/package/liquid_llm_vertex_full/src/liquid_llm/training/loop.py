import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from .metrics import cross_entropy, Meter
from .evaluation import evaluate


def _get_logits(output):
    """Handle both HF-style outputs with `.logits` and plain tensors."""
    if hasattr(output, "logits"):
        return output.logits
    return output


def _count_tokens(labels, fallback_tokens):
    """
    Prefer counting tokens actually contributing to loss (labels != -100).
    Fallback to total tokens if labels don't use ignore_index.
    """
    if labels is None:
        return fallback_tokens
    if labels.dtype == torch.long and (labels == -100).any():
        return (labels != -100).sum().item()
    return fallback_tokens


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
    teacher_name  = state["teacher_name"]
    kd_alpha      = state.get("kd_alpha", 0.5)
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
    hf_token      = state.get("hf_token")

    # --- Time-based checkpoint cadence & retention ---
    time_ckpt_secs            = int(state.get("time_ckpt_secs", 1800))            # every 30 min by default
    time_ckpt_retention_secs  = state.get("time_ckpt_retention_secs", 14400)      # 4 hours by default
    time_ckpt_keep_k          = state.get("time_ckpt_keep_k", None)               # default: count-unbounded
    if time_ckpt_retention_secs is not None:
        time_ckpt_retention_secs = int(time_ckpt_retention_secs)
    last_time_ckpt_ts         = float(state.get("last_time_ckpt_ts", 0.0))

    # --- Best checkpoint retention controls (versioned snapshots) ---
    best_ckpt_keep_k          = int(state.get("best_ckpt_keep_k", 3))             # keep K versioned best files
    best_ckpt_retention_secs  = state.get("best_ckpt_retention_secs", None)       # optional age prune
    if best_ckpt_retention_secs is not None:
        best_ckpt_retention_secs = int(best_ckpt_retention_secs)

    # --- Step checkpoint retention controls ---
    step_ckpt_keep_k          = int(state.get("step_ckpt_keep_k", 5))             # default keep last 5 step ckpts
    step_ckpt_retention_secs  = state.get("step_ckpt_retention_secs", None)       # optional age prune
    if step_ckpt_retention_secs is not None:
        step_ckpt_retention_secs = int(step_ckpt_retention_secs)

    # Track best validation loss for "best.pt"
    best_val_loss = float(state.get("best_val_loss", float("inf")))
    best_path_local = None

    # Optional teacher (for KD)
    teacher = None
    teacher_metrics = None
    if kd_alpha and kd_alpha > 0:
        auth_kwargs = {"token": hf_token} if hf_token else {}
        log.info(f"Loading teacher model '{teacher_name}' for knowledge distillation.")
        try:
            teacher = AutoModelForCausalLM.from_pretrained(teacher_name, **auth_kwargs).to(device).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            log.info(f"Teacher model '{teacher_name}' loaded and frozen for KD.")

            try:
                teacher_metrics = evaluate(teacher, val_loader, device=device)
                teacher.eval()  # ensure eval mode retained after evaluation helper
                log.info(
                    "[teacher] val_loss=%.4f ppl=%.2f",
                    teacher_metrics["val_loss"],
                    teacher_metrics["val_ppl"],
                )
            except Exception as eval_exc:  # pragma: no cover - defensive logging
                log.warning(f"Teacher evaluation failed: {eval_exc}")
                teacher_metrics = None

        except Exception as exc:  # pragma: no cover - defensive logging
            log.warning(f"Failed to load teacher model '{teacher_name}': {exc}")
            teacher = None
            teacher_metrics = None

    if teacher_metrics is not None:
        state["teacher_metrics"] = teacher_metrics
        state.setdefault("log_state", {}).setdefault("teacher_eval", teacher_metrics)

    step = state.get("step", 0)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

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
            with torch.amp.autocast("cuda", enabled=(precision == "fp16")):
                logits_s = _get_logits(model(input_ids))
                loss_ce  = cross_entropy(logits_s, labels)

                if teacher is not None and kd_alpha > 0:
                    with torch.no_grad():
                        logits_t = _get_logits(teacher(input_ids))
                    T = 1.0
                    logprob_s = torch.nn.functional.log_softmax(logits_s / T, dim=-1)
                    prob_t    = torch.nn.functional.softmax(logits_t / T, dim=-1)
                    loss_kd   = torch.nn.functional.kl_div(
                        logprob_s, prob_t, reduction="batchmean"
                    ) * (T * T)
                    loss = (1 - kd_alpha) * loss_ce + kd_alpha * loss_kd
                else:
                    loss = loss_ce

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # ===== Logging & meters =====
            tokens_in_batch = input_ids.numel()
            contrib_tokens  = _count_tokens(labels, tokens_in_batch)

            tok_meter.update(loss.item(), k=contrib_tokens)
            batch_meter.update(loss.item(), k=input_ids.size(0))

            since_log_tokens   += contrib_tokens
            since_log_examples += input_ids.size(0)

            # Periodic training log
            if step % state["log_interval"] == 0:
                elapsed = max(time.time() - t0, 1e-9)
                tok_per_sec  = since_log_tokens / elapsed
                ex_per_sec   = since_log_examples / elapsed
                log.info(
                    f"step={step}/{total_steps} "
                    f"loss_tok_mean={tok_meter.avg:.4f} "
                    f"loss_batch_mean={batch_meter.avg:.4f} "
                    f"tok/s={tok_per_sec:.0f} "
                    f"ex/s={ex_per_sec:.1f}"
                )
                tok_meter   = Meter()
                batch_meter = Meter()
                since_log_tokens   = 0
                since_log_examples = 0
                t0 = time.time()

            # Scheduled eval + best checkpointing
            if step % eval_every == 0:
                model.eval()
                metrics = evaluate(model, val_loader, device=device)
                val_loss = float(metrics["val_loss"])
                teacher_msg = ""
                teacher_metrics = state.get("teacher_metrics")
                if teacher_metrics:
                    teacher_msg = (
                        f" teacher_val_loss={teacher_metrics['val_loss']:.4f}"
                        f" teacher_ppl={teacher_metrics['val_ppl']:.2f}"
                    )
                log.info(
                    f"[eval] step={step} val_loss={val_loss:.4f} ppl={metrics['val_ppl']:.2f}" + teacher_msg
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
    return step