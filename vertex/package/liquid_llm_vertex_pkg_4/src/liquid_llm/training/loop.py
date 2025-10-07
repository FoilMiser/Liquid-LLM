import time
import statistics
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoModelForCausalLM
from .metrics import cross_entropy, Meter
from .evaluation import evaluate


class LossAveraging(Enum):
    PER_TOKEN = "per_tok"
    PER_BATCH = "per_batch"


def _format_float(value: float) -> str:
    if value == 0:
        return "0.000"
    formatted = f"{value:.6g}"
    digit_count = sum(ch.isdigit() for ch in formatted)
    if digit_count < 3:
        formatted = f"{value:.3f}"
    return formatted


def _format_loss(value: float, averaging: LossAveraging) -> str:
    return f"{_format_float(value)} ({averaging.value})"


def log_stats(log, tag: str, prefix: Optional[str] = None, **stats):
    parts: list[str] = [f"[{tag}]"]
    if prefix:
        parts.append(prefix)
    for key, value in stats.items():
        if isinstance(value, float):
            parts.append(f"{key}={_format_float(value)}")
        else:
            parts.append(f"{key}={value}")
    log.info(" ".join(parts))


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
    log_interval  = state["log_interval"]
    block_size    = state.get("block_size")
    kd_temperature = float(state.get("kd_temperature", 1.0))

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

    def _cuda_mem_strings():
        if not torch.cuda.is_available():
            return "0.00GB", "0.00GB"
        try:
            device_idx = torch.cuda.current_device()
            alloc = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 3)
            return f"{alloc:.2f}GB", f"{reserved:.2f}GB"
        except Exception:
            return "N/A", "N/A"

    # Optional teacher (for KD)
    teacher = None
    teacher_metrics = None
    teacher_load_error = None
    teacher_eval_error = None
    if kd_alpha and kd_alpha > 0:
        auth_kwargs = {"token": hf_token} if hf_token else {}
        log.info(f"Loading teacher model '{teacher_name}' for knowledge distillation.")
        try:
            teacher = AutoModelForCausalLM.from_pretrained(teacher_name, **auth_kwargs).to(device).eval()
            params_list = list(teacher.parameters())
            for p in params_list:
                p.requires_grad_(False)
            first_param = params_list[0] if params_list else None
            teacher_device = str(first_param.device) if first_param is not None else str(device)
            teacher_dtype = str(first_param.dtype) if first_param is not None else "unknown"
            num_params = sum(int(p.numel()) for p in params_list)
            seq_len = block_size if block_size is not None else "n/a"
            batch_size = getattr(train_loader, "batch_size", None)
            if batch_size is None:
                batch_size = state.get("micro_batch", "n/a")
            log_stats(
                log,
                "KD",
                teacher=teacher_name,
                device=teacher_device,
                dtype=teacher_dtype,
                params=f"{num_params:,}",
                seq=seq_len,
                bs=batch_size,
                T=_format_float(kd_temperature),
                alpha=f"{kd_alpha:.2f}",
                kl_scale="T^2",
            )

            try:
                teacher_metrics = evaluate(teacher, val_loader, device=device)
                teacher.eval()  # ensure eval mode retained after evaluation helper
                log_stats(
                    log,
                    "eval",
                    step=0,
                    val_loss_student="—",
                    ppl_student="—",
                    val_loss_teacher=_format_float(teacher_metrics["val_loss"]),
                    ppl_teacher=_format_float(teacher_metrics["val_ppl"]),
                )
            except Exception as eval_exc:  # pragma: no cover - defensive logging
                teacher_eval_error = eval_exc
                log.error(f"Teacher evaluation failed: {eval_exc}")
                teacher_metrics = None

        except Exception as exc:  # pragma: no cover - defensive logging
            teacher_load_error = exc
            log.error(f"Failed to load teacher model '{teacher_name}': {exc}")
            teacher = None
            teacher_metrics = None

        if teacher is None:
            msg = (
                "Knowledge distillation was requested (kd_alpha>0) but the teacher model "
                f"'{teacher_name}' could not be loaded. Aborting training."
            )
            raise RuntimeError(msg) from teacher_load_error

        if teacher_metrics is None:
            msg = (
                "Knowledge distillation was requested (kd_alpha>0) but the teacher model "
                "failed during evaluation and cannot provide baseline metrics. Aborting training."
            )
            raise RuntimeError(msg) from teacher_eval_error

    kd_active = teacher is not None and kd_alpha > 0

    if teacher_metrics is not None:
        state["teacher_metrics"] = teacher_metrics
        state.setdefault("log_state", {}).setdefault("teacher_eval", teacher_metrics)

    step = state.get("step", 0)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

    # Meters for logging
    tok_meter          = Meter()  # token-weighted mean (comparable to eval CE)
    batch_meter        = Meter()  # batch-weighted mean (debug)
    ce_meter           = Meter()
    kd_meter           = Meter()
    divergence_meter   = Meter()
    logit_delta_meter  = Meter()

    distill_history = deque(maxlen=500)
    total_loss_history = deque(maxlen=1000)

    last_plateau_warn_step = None
    last_spike_warn_step = None
    last_teacher_var_warn_step = None
    last_student_var_warn_step = None

    since_log_tokens   = 0
    since_log_examples = 0
    t0 = time.time()
    start_time = t0  # for first time-based ckpt window

    last_teacher_stats: Optional[dict] = None
    last_student_stats: Optional[dict] = None
    last_top1_match = None
    last_grad_norm = None
    last_total_loss_value = None
    last_perf = {
        "t_batch": 0.0,
        "t_teacher": 0.0,
        "t_student": 0.0,
        "t_kd": 0.0,
    }

    cuda_alloc, cuda_reserved = _cuda_mem_strings()
    current_lr = optimizer.param_groups[0].get("lr", 0.0) if optimizer.param_groups else 0.0
    log_stats(
        log,
        "opt",
        step=0,
        lr=f"{current_lr:.6g}",
        grad_norm="—",
        cuda_mem_alloc=cuda_alloc,
        cuda_mem_reserved=cuda_reserved,
    )

    model.train()
    while step < total_steps:
        for batch in train_loader:
            step += 1
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels    = batch["labels"].to(device, non_blocking=True)

            step_start = time.time()
            teacher_time = 0.0
            kd_time = 0.0

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(precision == "fp16")):
                student_start = time.time()
                logits_s = _get_logits(model(input_ids))
                loss_ce  = cross_entropy(logits_s, labels)
                student_time = time.time() - student_start

                loss_kd = None
                reverse_kd = 0.0
                mse_logits = 0.0
                top1_match_value = None
                if kd_active:
                    teacher_t0 = time.time()
                    with torch.no_grad():
                        logits_t = _get_logits(teacher(input_ids))
                    teacher_time = time.time() - teacher_t0

                    if logits_t.shape != logits_s.shape:
                        raise RuntimeError(
                            "Teacher and student logits shapes do not match: "
                            f"teacher={tuple(logits_t.shape)} student={tuple(logits_s.shape)}"
                        )

                    with torch.no_grad():
                        teacher_logits_stats = {
                            "mean": float(logits_t.float().mean()),
                            "std": float(torch.std(logits_t.float(), unbiased=False)),
                            "max": float(logits_t.max()),
                            "min": float(logits_t.min()),
                        }
                        logits_s_detached = logits_s.detach()
                        student_logits_stats = {
                            "mean": float(logits_s_detached.float().mean()),
                            "std": float(torch.std(logits_s_detached.float(), unbiased=False)),
                            "max": float(logits_s_detached.max()),
                            "min": float(logits_s_detached.min()),
                        }
                        last_teacher_stats = teacher_logits_stats
                        last_student_stats = student_logits_stats

                        match_tensor = (
                            logits_s_detached.argmax(dim=-1) == logits_t.argmax(dim=-1)
                        ).float().mean()
                        top1_match_value = float(match_tensor.item())

                        if teacher_logits_stats["std"] < 1.0 and (
                            last_teacher_var_warn_step is None or step - last_teacher_var_warn_step >= log_interval
                        ):
                            log.warning(
                                f"[warn] step={step} teacher logits std low ({_format_float(teacher_logits_stats['std'])})"
                            )
                            last_teacher_var_warn_step = step

                        if student_logits_stats["std"] < 0.5 and (
                            last_student_var_warn_step is None or step - last_student_var_warn_step >= log_interval
                        ):
                            log.warning(
                                f"[warn] step={step} student logits std low ({_format_float(student_logits_stats['std'])})"
                            )
                            last_student_var_warn_step = step

                    kd_start = time.time()
                    logprob_s = torch.nn.functional.log_softmax(logits_s / kd_temperature, dim=-1)
                    prob_t    = torch.nn.functional.softmax(logits_t / kd_temperature, dim=-1)
                    loss_kd   = torch.nn.functional.kl_div(
                        logprob_s, prob_t, reduction="batchmean"
                    ) * (kd_temperature * kd_temperature)
                    kd_time = time.time() - kd_start

                    with torch.no_grad():
                        logprob_t = torch.nn.functional.log_softmax(logits_t / kd_temperature, dim=-1)
                        prob_s    = torch.nn.functional.softmax(logits_s_detached / kd_temperature, dim=-1)
                        reverse_kd = float(
                            torch.nn.functional.kl_div(
                                logprob_t, prob_s, reduction="batchmean"
                            )
                            * (kd_temperature * kd_temperature)
                        )
                        mse_logits = float(
                            torch.nn.functional.mse_loss(
                                logits_s, logits_t, reduction="mean"
                            )
                        )

                    loss = (1 - kd_alpha) * loss_ce + kd_alpha * loss_kd
                else:
                    loss = loss_ce

            loss_ce_value = float(loss_ce.detach())
            loss_kd_value = float(loss_kd.detach()) if loss_kd is not None else 0.0
            reverse_kd_value = float(reverse_kd)
            mse_logits_value = float(mse_logits)
            total_loss_value = float(loss.detach())

            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            last_grad_norm = float(grad_norm) if grad_norm is not None else None
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # ===== Logging & meters =====
            tokens_in_batch = input_ids.numel()
            contrib_tokens  = _count_tokens(labels, tokens_in_batch)

            tok_meter.update(loss.item(), k=contrib_tokens)
            batch_meter.update(loss.item(), k=input_ids.size(0))
            ce_meter.update(loss_ce_value, k=contrib_tokens)
            if kd_active:
                kd_meter.update(loss_kd_value, k=contrib_tokens)
                divergence_meter.update(reverse_kd_value, k=contrib_tokens)
                logit_delta_meter.update(mse_logits_value, k=contrib_tokens)
                distill_history.append(loss_kd_value)
                if top1_match_value is not None:
                    last_top1_match = top1_match_value

            total_loss_history.append(total_loss_value)
            last_total_loss_value = total_loss_value

            since_log_tokens   += contrib_tokens
            since_log_examples += input_ids.size(0)

            step_time = time.time() - step_start
            last_perf = {
                "t_batch": step_time,
                "t_teacher": teacher_time,
                "t_student": student_time,
                "t_kd": kd_time,
            }

            # Periodic training log
            if step % log_interval == 0:
                elapsed = max(time.time() - t0, 1e-9)
                tok_per_sec  = since_log_tokens / elapsed
                ex_per_sec   = since_log_examples / elapsed

                cuda_alloc, cuda_reserved = _cuda_mem_strings()
                current_lr = (
                    optimizer.param_groups[0].get("lr", 0.0)
                    if optimizer.param_groups
                    else 0.0
                )
                grad_display = "—" if last_grad_norm is None else f"{last_grad_norm:.3f}"
                log_stats(
                    log,
                    "opt",
                    step=step,
                    lr=f"{current_lr:.6g}",
                    grad_norm=grad_display,
                    cuda_mem_alloc=cuda_alloc,
                    cuda_mem_reserved=cuda_reserved,
                )

                loss_entries = [
                    ("step", step),
                    ("student_loss", _format_loss(ce_meter.avg, LossAveraging.PER_TOKEN)),
                ]
                if kd_active and kd_meter.n:
                    loss_entries.append(
                        ("distill_loss", _format_loss(kd_meter.avg, LossAveraging.PER_TOKEN))
                    )
                loss_entries.append(
                    ("total_loss", _format_loss(tok_meter.avg, LossAveraging.PER_TOKEN))
                )
                if kd_active:
                    loss_entries.append(("alpha", f"{kd_alpha:.2f}"))
                    loss_entries.append(("T", _format_float(kd_temperature)))
                log_stats(log, "loss", **dict(loss_entries))

                if kd_active and kd_meter.n:
                    log_stats(
                        log,
                        "divergence",
                        kl_forward=kd_meter.avg,
                        kl_reverse=divergence_meter.avg,
                        mse=logit_delta_meter.avg,
                    )

                if last_teacher_stats:
                    log_stats(
                        log,
                        "logits",
                        prefix="teacher",
                        mean=last_teacher_stats["mean"],
                        std=last_teacher_stats["std"],
                        max=last_teacher_stats["max"],
                        min=last_teacher_stats["min"],
                    )
                if last_student_stats:
                    log_stats(
                        log,
                        "logits",
                        prefix="student",
                        mean=last_student_stats["mean"],
                        std=last_student_stats["std"],
                        max=last_student_stats["max"],
                        min=last_student_stats["min"],
                    )

                if kd_active and last_top1_match is not None:
                    log_stats(
                        log,
                        "align",
                        step=step,
                        top1_match=f"{last_top1_match:.3f}",
                    )

                perf_stats = {
                    "step": step,
                    "t_batch": f"{last_perf['t_batch']:.3f}s",
                    "t_teacher": f"{last_perf['t_teacher']:.3f}s",
                    "t_student": f"{last_perf['t_student']:.3f}s",
                    "t_kd": f"{last_perf['t_kd']:.3f}s",
                    "tok/s": f"{tok_per_sec:.0f}",
                    "ex/s": f"{ex_per_sec:.1f}",
                }
                log_stats(log, "perf", **perf_stats)

                if kd_active and len(distill_history) == distill_history.maxlen:
                    values = list(distill_history)
                    mid = len(values) // 2
                    first_med = statistics.median(values[:mid]) if mid else statistics.median(values)
                    second_med = statistics.median(values[mid:]) if mid else statistics.median(values)
                    reduction = (first_med - second_med) / max(abs(first_med), 1e-8)
                    if reduction < 0.01 and (
                        last_plateau_warn_step is None or step - last_plateau_warn_step >= log_interval
                    ):
                        log.warning(
                            f"[warn] step={step} distill_loss plateaued for {len(values)} steps (Δ < 1%)"
                        )
                        last_plateau_warn_step = step

                if total_loss_history:
                    best_window = min(total_loss_history)
                    if (
                        best_window > 0
                        and last_total_loss_value is not None
                        and last_total_loss_value > 1.5 * best_window
                        and (
                            last_spike_warn_step is None
                            or step - last_spike_warn_step >= log_interval
                        )
                    ):
                        log.warning(
                            "[warn] step=%s total_loss spiked (current=%.3f, best_window=%.3f)",
                            step,
                            last_total_loss_value,
                            best_window,
                        )
                        last_spike_warn_step = step

                tok_meter   = Meter()
                batch_meter = Meter()
                ce_meter           = Meter()
                kd_meter           = Meter()
                divergence_meter   = Meter()
                logit_delta_meter  = Meter()
                since_log_tokens   = 0
                since_log_examples = 0
                t0 = time.time()

            # Scheduled eval + best checkpointing
            if step % eval_every == 0:
                model.eval()
                metrics = evaluate(model, val_loader, device=device)
                val_loss = float(metrics["val_loss"])
                teacher_metrics = state.get("teacher_metrics")
                eval_stats = {
                    "step": step,
                    "val_loss_student": _format_float(val_loss),
                    "ppl_student": _format_float(metrics["val_ppl"]),
                }
                if teacher_metrics:
                    eval_stats["val_loss_teacher"] = _format_float(teacher_metrics["val_loss"])
                    eval_stats["ppl_teacher"] = _format_float(teacher_metrics["val_ppl"])
                log_stats(log, "eval", **eval_stats)

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