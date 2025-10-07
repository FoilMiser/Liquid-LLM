import time
import math
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoModelForCausalLM
from .metrics import cross_entropy, Meter
from .evaluation import evaluate

def _format_float(value: float) -> str:
    if value == 0:
        return "0.000"
    formatted = f"{value:.6g}"
    digit_count = sum(ch.isdigit() for ch in formatted)
    if digit_count < 3:
        formatted = f"{value:.3f}"
    return formatted


def _update_ema(current: Optional[float], value: float, beta: float) -> float:
    if current is None:
        return value
    return beta * current + (1.0 - beta) * value


def _token_logprob_corr(logprob_t: torch.Tensor, logprob_s: torch.Tensor) -> float:
    if logprob_t.numel() == 0 or logprob_s.numel() == 0:
        return 0.0
    x = logprob_t.float().view(-1, logprob_t.size(-1))
    y = logprob_s.float().view(-1, logprob_s.size(-1))
    if x.size(0) == 0:
        return 0.0
    x_center = x - x.mean(dim=1, keepdim=True)
    y_center = y - y.mean(dim=1, keepdim=True)
    cov = (x_center * y_center).sum(dim=1)
    x_var = x_center.pow(2).sum(dim=1)
    y_var = y_center.pow(2).sum(dim=1)
    denom = torch.sqrt(x_var * y_var + 1e-8)
    corr = torch.where(denom > 0, cov / denom, torch.zeros_like(cov))
    return float(torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).mean().item())


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
    """Helper to save a checkpoint and log locations.

    Returns
    -------
    tuple[str, str | None, float, int]
        Tuple of (local path, optional uploaded URI, elapsed seconds, file size bytes).
    """
    start = time.time()
    state_dict = {
        "model": state["model"].state_dict(),
        "optimizer": state["optimizer"].state_dict(),
        "scheduler": state["scheduler"].state_dict(),
        "step": step,
    }
    local, uri = ckptio(
        state_dict,
        state["local_outdir"],
        state.get("gcs_outdir"),
        filename=filename,
    )
    elapsed = time.time() - start
    try:
        size_bytes = Path(local).stat().st_size
    except OSError:
        size_bytes = 0
    return local, uri, elapsed, size_bytes


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

    return survivors, removed


def _prune_time_ckpts(local_outdir: str, log, retention_secs: int | None, keep_k: int | None, now: float | None = None):
    """
    Prune *time-based* checkpoints matching `ckpt_time_*.pt`.
    """
    now = now or time.time()
    files = _list_sorted_by_mtime("ckpt_time_*.pt", local_outdir)
    _, removed = _prune_files(
        files,
        keep_k=keep_k,
        retention_secs=retention_secs,
        now=now,
        log=log,
        label="time-based",
    )
    return removed


def _prune_best_ckpts(local_outdir: str, log, retention_secs: int | None, keep_k: int | None, now: float | None = None):
    """
    Prune versioned best checkpoints saved as `ckpt_best_*.pt`.
    NOTE: Does NOT remove the rolling `best.pt`.
    """
    now = now or time.time()
    files = _list_sorted_by_mtime("ckpt_best_*.pt", local_outdir)
    if not files:
        return 0
    _, removed = _prune_files(
        files,
        keep_k=keep_k,
        retention_secs=retention_secs,
        now=now,
        log=log,
        label="best",
    )
    return removed


def _prune_step_ckpts(local_outdir: str, log, retention_secs: int | None, keep_k: int | None, now: float | None = None):
    """
    Prune step-based checkpoints saved as `ckpt_step_*.pt`.
    """
    now = now or time.time()
    files = _list_sorted_by_mtime("ckpt_step_*.pt", local_outdir)
    if not files:
        return 0
    _, removed = _prune_files(
        files,
        keep_k=keep_k,
        retention_secs=retention_secs,
        now=now,
        log=log,
        label="step-based",
    )
    return removed


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
    pad_id        = state.get("pad_id")
    grad_clip_norm = float(state.get("grad_clip_norm", 1.0))
    loss_ema_beta = float(state.get("loss_ema_beta", 0.9))
    dataset_name = state.get("dataset_name", "unknown")
    val_split_name = state.get("val_split", "validation")
    seed_value = state.get("seed", "unknown")
    run_id = state.get("run_id", "unknown")
    run_meta = state.get("run_meta", {})
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            world_size = torch.distributed.get_world_size()
        except Exception:
            world_size = 1
    global_batch = state.get("global_batch") or getattr(train_loader, "batch_size", None)
    if global_batch is None:
        global_batch = state.get("micro_batch", "n/a")
    seq_len = block_size if block_size is not None else "n/a"

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
    teacher_load_error = None
    teacher_eval_error = None
    if kd_alpha and kd_alpha > 0:
        auth_kwargs = {"token": hf_token} if hf_token else {}
        try:
            teacher = AutoModelForCausalLM.from_pretrained(teacher_name, **auth_kwargs).to(device).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

            try:
                teacher_eval_t0 = time.time()
                teacher_metrics = evaluate(teacher, val_loader, device=device)
                teacher_eval_time = time.time() - teacher_eval_t0
                teacher.eval()  # ensure eval mode retained after evaluation helper
                log_stats(
                    log,
                    "eval",
                    step=0,
                    dataset=dataset_name,
                    split=val_split_name,
                    tokens=teacher_metrics.get("tokens", "n/a"),
                    val_loss_student="na",
                    ppl_student="na",
                    val_loss_teacher=_format_float(teacher_metrics["val_loss"]),
                    ppl_teacher=_format_float(teacher_metrics["val_ppl"]),
                    t_eval=_format_float(teacher_eval_time),
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
    initial_kd_scheme = "forward_kl" if kd_active else "off"

    run_fields = {
        "run_id": run_id,
        "git_sha": run_meta.get("git_sha", "unknown"),
        "pkg_ver": run_meta.get("pkg_ver", "unknown"),
        "seed": seed_value,
        "device": device,
        "world_size": world_size,
        "precision": precision,
        "dataset": dataset_name,
        "seq_len": seq_len,
        "batch": global_batch,
        "teacher": teacher_name,
        "alpha": _format_float(kd_alpha),
        "T": _format_float(kd_temperature),
        "kd_scheme": initial_kd_scheme,
        "kl_scale": "T^2",
    }
    log_stats(log, "run", **run_fields)

    if teacher_metrics is not None:
        state["teacher_metrics"] = teacher_metrics
        state.setdefault("log_state", {}).setdefault("teacher_eval", teacher_metrics)

    step = state.get("step", 0)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

    kd_params_last = None

    def maybe_log_kd_params(step_idx: int):
        nonlocal kd_params_last
        current_kd_alpha = float(state.get("kd_alpha", kd_alpha))
        current_kd_temperature = float(state.get("kd_temperature", kd_temperature))
        scheme = "forward_kl" if (teacher is not None and current_kd_alpha > 0) else "off"
        current = (current_kd_alpha, current_kd_temperature, scheme)
        if current != kd_params_last:
            log_stats(
                log,
                "kd-hparams",
                step=step_idx,
                alpha=_format_float(current_kd_alpha),
                T=_format_float(current_kd_temperature),
                scheme=scheme,
            )
            kd_params_last = current

    maybe_log_kd_params(step)

    # Meters for logging
    tok_meter = Meter()
    ce_tok_meter = Meter()
    kd_tok_meter = Meter()
    divergence_meter = Meter()
    logit_delta_meter = Meter()
    entropy_student_meter = Meter()
    entropy_teacher_meter = Meter()

    since_log_tokens = 0
    since_log_total_tokens = 0
    since_log_examples = 0
    since_log_steps = 0
    pad_tokens_since_log = 0
    attn_mask_ones_since_log = 0
    attn_mask_elems_since_log = 0
    t0 = time.time()
    start_time = t0  # for first time-based ckpt window
    step_time_accum = 0.0
    ckpt_time_accum = 0.0
    watchdog_interval = 100
    grad_clipped_count_cum = 0
    nan_inf_since_watchdog = False

    tokens_seen_total = int(state.get("tokens_seen_total", 0))
    examples_seen_total = int(state.get("examples_seen_total", 0))
    ema_ce_tok = state.get("ema_ce_tok")
    ema_kd_tok = state.get("ema_kd_tok")
    ema_total_tok = state.get("ema_total_tok")

    last_top1_match = None
    last_top5_match = None
    last_logprob_corr = None
    last_grad_norm = None
    last_clip_triggered = False

    model.train()
    train_iter = iter(train_loader)

    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        step += 1
        since_log_steps += 1

        current_kd_alpha = float(state.get("kd_alpha", kd_alpha))
        current_kd_temperature = float(state.get("kd_temperature", kd_temperature))
        kd_active_step = teacher is not None and current_kd_alpha > 0
        kd_scheme_step = "forward_kl" if kd_active_step else "off"
        maybe_log_kd_params(step)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attn_mask_ones_since_log += int(attention_mask.long().sum().item())
            attn_mask_elems_since_log += attention_mask.numel()

        tokens_in_batch_total = input_ids.numel()
        since_log_total_tokens += tokens_in_batch_total
        if pad_id is not None:
            pad_tokens_since_log += int((input_ids == pad_id).sum().item())

        step_start = time.time()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(precision == "fp16")):
            logits_s = _get_logits(model(input_ids))
            loss_ce = cross_entropy(logits_s, labels)

            logits_s_detached = logits_s.detach()

            loss_kd = None
            reverse_kd = 0.0
            mse_logits = 0.0
            top1_match_value = None
            top5_match_value = None
            logprob_corr_value = None
            teacher_entropy_sum = 0.0

            if kd_active_step:
                with torch.no_grad():
                    logits_t = _get_logits(teacher(input_ids))

                if logits_t.shape != logits_s.shape:
                    raise RuntimeError(
                        "Teacher and student logits shapes do not match: "
                        f"teacher={tuple(logits_t.shape)} student={tuple(logits_s.shape)}"
                    )

                with torch.no_grad():
                    match_tensor = (
                        logits_s_detached.argmax(dim=-1) == logits_t.argmax(dim=-1)
                    ).float().mean()
                    top1_match_value = float(match_tensor.item())

                    k_top = min(5, logits_t.size(-1))
                    if k_top > 0:
                        student_topk = torch.topk(logits_s_detached, k=k_top, dim=-1).indices
                        teacher_topk = torch.topk(logits_t, k=k_top, dim=-1).indices
                        overlap = torch.isin(teacher_topk, student_topk)
                        top5_match_value = float(overlap.float().mean().item())

                logprob_s_temp = torch.nn.functional.log_softmax(
                    logits_s / current_kd_temperature, dim=-1
                )
                prob_t = torch.nn.functional.softmax(
                    logits_t / current_kd_temperature, dim=-1
                )
                loss_kd = torch.nn.functional.kl_div(
                    logprob_s_temp, prob_t, reduction="batchmean"
                ) * (current_kd_temperature * current_kd_temperature)

                with torch.no_grad():
                    logprob_t_temp = torch.nn.functional.log_softmax(
                        logits_t / current_kd_temperature, dim=-1
                    )
                    prob_s = torch.nn.functional.softmax(
                        logits_s_detached / current_kd_temperature, dim=-1
                    )
                    reverse_kd = float(
                        torch.nn.functional.kl_div(
                            logprob_t_temp, prob_s, reduction="batchmean"
                        )
                        * (current_kd_temperature * current_kd_temperature)
                    )
                    mse_logits = float(
                        torch.nn.functional.mse_loss(logits_s, logits_t, reduction="mean")
                    )
                    logprob_corr_value = _token_logprob_corr(
                        logprob_t_temp, logprob_s_temp.detach()
                    )

                    teacher_logprob_base = torch.nn.functional.log_softmax(
                        logits_t.float(), dim=-1
                    )
                    teacher_prob_base = teacher_logprob_base.exp()
                    teacher_entropy_sum = float(
                        (-(teacher_prob_base * teacher_logprob_base).sum(dim=-1)).sum().item()
                    )

                loss = (1 - current_kd_alpha) * loss_ce + current_kd_alpha * loss_kd
            else:
                loss = loss_ce

            with torch.no_grad():
                student_logprob_base = torch.nn.functional.log_softmax(
                    logits_s_detached.float(), dim=-1
                )
                student_prob_base = student_logprob_base.exp()
                student_entropy_sum = float(
                    (-(student_prob_base * student_logprob_base).sum(dim=-1)).sum().item()
                )

        loss_ce_value = float(loss_ce.detach())
        loss_kd_value = float(loss_kd.detach()) if loss_kd is not None else 0.0
        reverse_kd_value = float(reverse_kd)
        mse_logits_value = float(mse_logits)
        total_loss_value = float(loss.detach())
        if (
            not math.isfinite(total_loss_value)
            or not math.isfinite(loss_ce_value)
            or (kd_active_step and loss_kd is not None and not math.isfinite(loss_kd_value))
        ):
            nan_inf_since_watchdog = True

        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        last_grad_norm = float(grad_norm) if grad_norm is not None else None
        last_clip_triggered = bool(
            grad_norm is not None and last_grad_norm is not None and last_grad_norm > grad_clip_norm + 1e-6
        )
        if last_grad_norm is not None and not math.isfinite(last_grad_norm):
            nan_inf_since_watchdog = True
        if last_clip_triggered:
            grad_clipped_count_cum += 1
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        tokens_in_batch = input_ids.numel()
        contrib_tokens = _count_tokens(labels, tokens_in_batch)
        batch_size = input_ids.size(0)

        tok_meter.update(total_loss_value, k=contrib_tokens)
        ce_tok_meter.update(loss_ce_value, k=contrib_tokens)
        if kd_active_step and loss_kd is not None:
            kd_tok_meter.update(loss_kd_value, k=contrib_tokens)
            divergence_meter.update(reverse_kd_value, k=contrib_tokens)
            logit_delta_meter.update(mse_logits_value, k=contrib_tokens)
        else:
            ema_kd_tok = None
        if logprob_corr_value is not None:
            last_logprob_corr = logprob_corr_value
        if top1_match_value is not None:
            last_top1_match = top1_match_value
        if top5_match_value is not None:
            last_top5_match = top5_match_value

        entropy_student_meter.update(
            student_entropy_sum / max(contrib_tokens, 1), k=contrib_tokens
        )
        if kd_active_step and loss_kd is not None:
            entropy_teacher_meter.update(
                teacher_entropy_sum / max(contrib_tokens, 1), k=contrib_tokens
            )

        since_log_tokens += contrib_tokens
        since_log_examples += batch_size
        tokens_seen_total += contrib_tokens
        examples_seen_total += batch_size

        ema_ce_tok = _update_ema(ema_ce_tok, loss_ce_value, loss_ema_beta)
        ema_total_tok = _update_ema(ema_total_tok, total_loss_value, loss_ema_beta)
        if kd_active_step and loss_kd is not None:
            ema_kd_tok = _update_ema(ema_kd_tok, loss_kd_value, loss_ema_beta)

        step_time = time.time() - step_start
        step_time_accum += step_time

        if step % log_interval == 0:
            elapsed = max(time.time() - t0, 1e-9)
            tok_per_sec = since_log_tokens / elapsed
            ex_per_sec = since_log_examples / max(elapsed, 1e-9)
            avg_step = step_time_accum / max(since_log_steps, 1)
            pad_frac = pad_tokens_since_log / max(since_log_total_tokens, 1)
            attn_cov = (
                attn_mask_ones_since_log / max(attn_mask_elems_since_log, 1)
                if attn_mask_elems_since_log
                else None
            )
            current_lr = (
                optimizer.param_groups[0].get("lr", 0.0)
                if optimizer.param_groups
                else 0.0
            )
            grad_display = "na" if last_grad_norm is None else _format_float(last_grad_norm)
            clipped_display = "YES" if last_clip_triggered else "NO"
            ce_tok_avg = ce_tok_meter.avg if ce_tok_meter.n else 0.0
            kd_tok_avg = kd_tok_meter.avg if kd_tok_meter.n else 0.0
            total_tok_avg = tok_meter.avg if tok_meter.n else 0.0
            kl_rev_avg = divergence_meter.avg if divergence_meter.n else None
            mse_avg = logit_delta_meter.avg if logit_delta_meter.n else None
            H_student_avg = entropy_student_meter.avg if entropy_student_meter.n else 0.0
            H_teacher_avg = entropy_teacher_meter.avg if entropy_teacher_meter.n else None
            corr_value = last_logprob_corr
            top1_value = last_top1_match
            top5_value = last_top5_match
            eff_kd = current_kd_alpha * (current_kd_temperature * current_kd_temperature)

            train_stats = {
                "step": step,
                "tokens_seen": tokens_seen_total,
                "lr": _format_float(current_lr),
                "grad_norm": grad_display,
                "clipped": clipped_display,
                "ce_tok": _format_float(ce_tok_avg),
                "kd_tok": _format_float(kd_tok_avg) if kd_tok_meter.n else "na",
                "total_tok": _format_float(total_tok_avg),
                "alpha": _format_float(current_kd_alpha),
                "T": _format_float(current_kd_temperature),
                "eff_kd": _format_float(eff_kd),
                "kl_fwd": _format_float(kd_tok_avg) if kd_tok_meter.n else "na",
                "kl_rev": _format_float(kl_rev_avg) if kl_rev_avg is not None else "na",
                "mse": _format_float(mse_avg) if mse_avg is not None else "na",
                "H_student": _format_float(H_student_avg),
                "H_teacher": _format_float(H_teacher_avg) if H_teacher_avg is not None else "na",
                "top1": _format_float(top1_value) if top1_value is not None else "na",
                "top5": _format_float(top5_value) if top5_value is not None else "na",
                "corr_t_s": _format_float(corr_value) if corr_value is not None else "na",
                "pad_frac": _format_float(pad_frac),
                "attn_mask_coverage": _format_float(attn_cov) if attn_cov is not None else "na",
                "t_step": _format_float(avg_step),
                "tok/s": _format_float(tok_per_sec),
                "ex/s": _format_float(ex_per_sec),
            }
            log_stats(log, "train", **train_stats)

            tok_meter = Meter()
            ce_tok_meter = Meter()
            kd_tok_meter = Meter()
            divergence_meter = Meter()
            logit_delta_meter = Meter()
            entropy_student_meter = Meter()
            entropy_teacher_meter = Meter()
            since_log_tokens = 0
            since_log_total_tokens = 0
            since_log_examples = 0
            since_log_steps = 0
            pad_tokens_since_log = 0
            attn_mask_ones_since_log = 0
            attn_mask_elems_since_log = 0
            step_time_accum = 0.0
            t0 = time.time()

        if step % watchdog_interval == 0:
            log_stats(
                log,
                "watchdog",
                step=step,
                nan_inf_found="YES" if nan_inf_since_watchdog else "NO",
                grad_clipped_count_cum=grad_clipped_count_cum,
            )
            nan_inf_since_watchdog = False

        kd_alpha = current_kd_alpha
        kd_temperature = current_kd_temperature

        if step >= total_steps:
            break
        # Scheduled eval + best checkpointing
        if step % eval_every == 0:
            model.eval()
            eval_t0 = time.time()
            metrics = evaluate(model, val_loader, device=device)
            eval_time = time.time() - eval_t0
            val_loss = float(metrics["val_loss"])
            teacher_metrics = state.get("teacher_metrics")
            eval_stats = {
                "step": step,
                "dataset": dataset_name,
                "split": val_split_name,
                "tokens": metrics.get("tokens", "n/a"),
                "val_loss_student": _format_float(val_loss),
                "ppl_student": _format_float(metrics["val_ppl"]),
                "val_loss_teacher": _format_float(teacher_metrics["val_loss"]) if teacher_metrics else "na",
                "ppl_teacher": _format_float(teacher_metrics["val_ppl"]) if teacher_metrics else "na",
                "t_eval": _format_float(eval_time),
            }
            log_stats(log, "eval", **eval_stats)

            # New best? Save/overwrite best.pt and versioned best snapshot
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                # Rolling "best.pt" (always the current best)
                best_path_local, best_uri, best_time, best_size = _save_ckpt(
                    ckptio, state, step, filename="best.pt", log=log
                )
                ckpt_time_accum += best_time
                log_state = state.setdefault("log_state", {})
                if state.get("gcs_outdir") and not best_uri:
                    log.warning(
                        "[ckpt] expected to upload best checkpoint to %s but no URI was returned",
                        state["gcs_outdir"],
                    )
                log_state["best_checkpoint"] = {
                    "step": step,
                    "val_loss": float(val_loss),
                    "local_path": best_path_local,
                    "gcs_uri": best_uri,
                }
                log_state.setdefault("best_history", []).append(
                    {
                        "step": step,
                        "val_loss": float(val_loss),
                        "local_path": best_path_local,
                        "gcs_uri": best_uri,
                    }
                )

                log_stats(
                    log,
                    "ckpt",
                    status="new_best",
                    step=step,
                    val_loss_student=_format_float(val_loss),
                    filename="best.pt",
                    pruned_count=0,
                )

                # Versioned best snapshot for historical/top-K retention
                vers_name = f"ckpt_best_step{step}_vl{val_loss:.4f}.pt"
                vers_local, vers_uri, vers_time, vers_size = _save_ckpt(
                    ckptio, state, step, filename=vers_name, log=log
                )
                ckpt_time_accum += vers_time
                log_state.setdefault("best_versioned", []).append(
                    {
                        "step": step,
                        "val_loss": float(val_loss),
                        "local_path": vers_local,
                        "gcs_uri": vers_uri,
                    }
                )

                log_stats(
                    log,
                    "ckpt",
                    status="best_versioned",
                    step=step,
                    val_loss_student=_format_float(val_loss),
                    filename=vers_name,
                    pruned_count=0,
                )

                # Prune older best checkpoints per policy
                prune_t0 = time.time()
                pruned_best_count = 0
                try:
                    pruned_best_count = _prune_best_ckpts(
                        local_outdir=state["local_outdir"],
                        log=log,
                        keep_k=best_ckpt_keep_k,
                        retention_secs=best_ckpt_retention_secs,
                        now=time.time(),
                    )
                    prune_time = time.time() - prune_t0
                    ckpt_time_accum += prune_time
                except Exception as e:
                    ckpt_time_accum += time.time() - prune_t0
                    log.warning(f"[ckpt] best prune error: {e}")
                if pruned_best_count:
                    log_stats(
                        log,
                        "ckpt",
                        status="best_prune",
                        step=step,
                        val_loss_student=_format_float(val_loss),
                        filename=vers_name,
                        pruned_count=pruned_best_count,
                    )

            model.train()

            # Step-based checkpointing
            if save_every > 0 and step % save_every == 0:
                step_filename = f"ckpt_step_{step}.pt"
                step_local, step_uri, step_ckpt_time, step_size = _save_ckpt(
                    ckptio, state, step, filename=step_filename, log=log
                )
                ckpt_time_accum += step_ckpt_time
                state.setdefault("log_state", {}).setdefault("step_checkpoints", []).append(
                    {
                        "step": step,
                        "local_path": step_local,
                        "gcs_uri": step_uri,
                    }
                )
                log_stats(
                    log,
                    "ckpt",
                    status="step",
                    step=step,
                    val_loss_student="na",
                    filename=step_filename,
                    pruned_count=0,
                )
                # Prune older step checkpoints per policy (age &/or count)
                prune_t0 = time.time()
                try:
                    pruned_step_count = _prune_step_ckpts(
                        local_outdir=state["local_outdir"],
                        log=log,
                        retention_secs=step_ckpt_retention_secs,
                        keep_k=step_ckpt_keep_k,
                        now=time.time(),
                    )
                    prune_time = time.time() - prune_t0
                    ckpt_time_accum += prune_time
                    if pruned_step_count:
                        log_stats(
                            log,
                            "ckpt",
                            status="step_prune",
                            step=step,
                            val_loss_student="na",
                            filename=step_filename,
                            pruned_count=pruned_step_count,
                        )
                except Exception as e:
                    ckpt_time_accum += time.time() - prune_t0
                    log.warning(f"[ckpt] step prune error: {e}")

            # Time-based checkpointing (e.g., every 30–60 minutes)
            now_ts = time.time()
            if last_time_ckpt_ts == 0.0:
                last_time_ckpt_ts = start_time
            if time_ckpt_secs > 0 and (now_ts - last_time_ckpt_ts) >= time_ckpt_secs:
                ts = int(now_ts)
                fname = f"ckpt_time_{ts}.pt"
                time_local, time_uri, time_elapsed, time_size = _save_ckpt(
                    ckptio, state, step, filename=fname, log=log
                )
                ckpt_time_accum += time_elapsed
                state.setdefault("log_state", {}).setdefault("time_checkpoints", []).append(
                    {
                        "step": step,
                        "local_path": time_local,
                        "gcs_uri": time_uri,
                    }
                )
                log_stats(
                    log,
                    "ckpt",
                    status="time",
                    step=step,
                    val_loss_student="na",
                    filename=fname,
                    pruned_count=0,
                )
                last_time_ckpt_ts = now_ts
                state["last_time_ckpt_ts"] = last_time_ckpt_ts  # persist within this run

                # Prune older time-based checkpoints to save disk
                prune_t0 = time.time()
                try:
                    pruned_time_count = _prune_time_ckpts(
                        local_outdir=state["local_outdir"],
                        log=log,
                        retention_secs=time_ckpt_retention_secs,
                        keep_k=time_ckpt_keep_k,
                        now=now_ts,
                    )
                    prune_time = time.time() - prune_t0
                    ckpt_time_accum += prune_time
                    if pruned_time_count:
                        log_stats(
                            log,
                            "ckpt",
                            status="time_prune",
                            step=step,
                            val_loss_student="na",
                            filename=fname,
                            pruned_count=pruned_time_count,
                        )
                except Exception as e:
                    ckpt_time_accum += time.time() - prune_t0
                    log.warning(f"[ckpt] time prune error: {e}")

            if step >= total_steps:
                break

    # Persist best loss and running stats to state (helpful if caller saves run metadata)
    state["best_val_loss"] = best_val_loss
    state["tokens_seen_total"] = tokens_seen_total
    state["examples_seen_total"] = examples_seen_total
    state["ema_ce_tok"] = ema_ce_tok
    state["ema_kd_tok"] = ema_kd_tok
    state["ema_total_tok"] = ema_total_tok
    return step