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


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(max(num_bytes, 0))
    unit = units[0]
    for u in units[1:]:
        if size < 1024.0:
            break
        size /= 1024.0
        unit = u
    return f"{size:.2f}{unit}"


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
    pad_id        = state.get("pad_id")
    grad_clip_norm = float(state.get("grad_clip_norm", 1.0))
    loss_ema_beta = float(state.get("loss_ema_beta", 0.9))
    ema_window = max(min(int(round(1.0 / max(1.0 - loss_ema_beta, 1e-6))), 10000), 1)
    dataset_name = state.get("dataset_name", "unknown")
    val_split_name = state.get("val_split", "validation")

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
    tok_meter = Meter()
    batch_meter = Meter()
    ce_tok_meter = Meter()
    ce_seq_meter = Meter()
    kd_tok_meter = Meter()
    kd_seq_meter = Meter()
    total_seq_meter = Meter()
    divergence_meter = Meter()
    logit_delta_meter = Meter()
    entropy_student_meter = Meter()
    entropy_teacher_meter = Meter()

    distill_history = deque(maxlen=500)
    total_loss_history = deque(maxlen=1000)

    last_plateau_warn_step = None
    last_spike_warn_step = None
    last_teacher_var_warn_step = None
    last_student_var_warn_step = None

    since_log_tokens = 0
    since_log_total_tokens = 0
    since_log_examples = 0
    since_log_steps = 0
    pad_tokens_since_log = 0
    attn_mask_ones_since_log = 0
    attn_mask_elems_since_log = 0
    t0 = time.time()
    start_time = t0  # for first time-based ckpt window

    tokens_seen_total = int(state.get("tokens_seen_total", 0))
    examples_seen_total = int(state.get("examples_seen_total", 0))
    ema_ce_tok = state.get("ema_ce_tok")
    ema_kd_tok = state.get("ema_kd_tok")
    ema_total_tok = state.get("ema_total_tok")

    last_teacher_stats: Optional[dict] = None
    last_student_stats: Optional[dict] = None
    last_top1_match = None
    last_top5_match = None
    last_logprob_corr = None
    last_grad_norm = None
    last_clip_triggered = False
    last_total_loss_value = None

    perf_totals = {
        "loader": 0.0,
        "student": 0.0,
        "teacher": 0.0,
        "kd": 0.0,
        "step": 0.0,
    }
    ckpt_time_accum = 0.0

    cuda_alloc, cuda_reserved = _cuda_mem_strings()
    current_lr = optimizer.param_groups[0].get("lr", 0.0) if optimizer.param_groups else 0.0
    log_stats(
        log,
        "opt",
        step=0,
        lr=f"{current_lr:.6g}",
        grad_norm="—",
        clipped=f"NO({_format_float(grad_clip_norm)})",
        amp_scale=f"{scaler.get_scale():.3f}",
        cuda_mem_alloc=cuda_alloc,
        cuda_mem_reserved=cuda_reserved,
    )

    kd_scheme = "forward_kl" if kd_active else "off"
    kd_params_last = (float(kd_alpha), float(kd_temperature), kd_scheme)
    log_stats(
        log,
        "kd-hparams",
        alpha=_format_float(kd_alpha),
        T=_format_float(kd_temperature),
        scheme=kd_scheme,
    )

    model.train()
    train_iter = iter(train_loader)

    while step < total_steps:
        loader_t0 = time.time()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            loader_t0 = time.time()
            batch = next(train_iter)
        loader_time = time.time() - loader_t0
        perf_totals["loader"] += loader_time

        step += 1
        since_log_steps += 1

        current_kd_alpha = float(state.get("kd_alpha", kd_alpha))
        current_kd_temperature = float(state.get("kd_temperature", kd_temperature))
        kd_active_step = teacher is not None and current_kd_alpha > 0
        kd_scheme_step = "forward_kl" if kd_active_step else "off"
        current_params = (float(current_kd_alpha), float(current_kd_temperature), kd_scheme_step)
        if current_params != kd_params_last:
            log_stats(
                log,
                "kd-hparams",
                step=step,
                alpha=_format_float(current_kd_alpha),
                T=_format_float(current_kd_temperature),
                scheme=kd_scheme_step,
            )
            kd_params_last = current_params

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
        teacher_time = 0.0
        kd_time = 0.0

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(precision == "fp16")):
            student_start = time.time()
            logits_s = _get_logits(model(input_ids))
            loss_ce = cross_entropy(logits_s, labels)
            student_time = time.time() - student_start
            perf_totals["student"] += student_time

            logits_s_detached = logits_s.detach()
            student_logits_stats = {
                "mean": float(logits_s_detached.float().mean()),
                "std": float(torch.std(logits_s_detached.float(), unbiased=False)),
                "max": float(logits_s_detached.max()),
                "min": float(logits_s_detached.min()),
            }
            last_student_stats = student_logits_stats

            loss_kd = None
            reverse_kd = 0.0
            mse_logits = 0.0
            top1_match_value = None
            top5_match_value = None
            logprob_corr_value = None
            teacher_entropy_sum = 0.0

            if kd_active_step:
                teacher_t0 = time.time()
                with torch.no_grad():
                    logits_t = _get_logits(teacher(input_ids))
                teacher_time = time.time() - teacher_t0
                perf_totals["teacher"] += teacher_time

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
                    last_teacher_stats = teacher_logits_stats

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
                logprob_s_temp = torch.nn.functional.log_softmax(
                    logits_s / current_kd_temperature, dim=-1
                )
                prob_t = torch.nn.functional.softmax(
                    logits_t / current_kd_temperature, dim=-1
                )
                loss_kd = torch.nn.functional.kl_div(
                    logprob_s_temp, prob_t, reduction="batchmean"
                ) * (current_kd_temperature * current_kd_temperature)
                kd_time = time.time() - kd_start
                perf_totals["kd"] += kd_time

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

        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        last_grad_norm = float(grad_norm) if grad_norm is not None else None
        last_clip_triggered = bool(
            grad_norm is not None and last_grad_norm is not None and last_grad_norm > grad_clip_norm + 1e-6
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        amp_scale_value = scaler.get_scale()

        tokens_in_batch = input_ids.numel()
        contrib_tokens = _count_tokens(labels, tokens_in_batch)
        batch_size = input_ids.size(0)

        tok_meter.update(total_loss_value, k=contrib_tokens)
        batch_meter.update(total_loss_value, k=batch_size)
        ce_tok_meter.update(loss_ce_value, k=contrib_tokens)
        ce_seq_meter.update(
            loss_ce_value * contrib_tokens / max(batch_size, 1), k=batch_size
        )
        total_seq_meter.update(
            total_loss_value * contrib_tokens / max(batch_size, 1), k=batch_size
        )
        if kd_active_step and loss_kd is not None:
            kd_tok_meter.update(loss_kd_value, k=contrib_tokens)
            kd_seq_meter.update(
                loss_kd_value * contrib_tokens / max(batch_size, 1), k=batch_size
            )
            divergence_meter.update(reverse_kd_value, k=contrib_tokens)
            logit_delta_meter.update(mse_logits_value, k=contrib_tokens)
            distill_history.append(loss_kd_value)
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

        total_loss_history.append(total_loss_value)
        last_total_loss_value = total_loss_value

        since_log_tokens += contrib_tokens
        since_log_examples += batch_size
        tokens_seen_total += contrib_tokens
        examples_seen_total += batch_size

        ema_ce_tok = _update_ema(ema_ce_tok, loss_ce_value, loss_ema_beta)
        ema_total_tok = _update_ema(ema_total_tok, total_loss_value, loss_ema_beta)
        if kd_active_step and loss_kd is not None:
            ema_kd_tok = _update_ema(ema_kd_tok, loss_kd_value, loss_ema_beta)

        step_time = time.time() - step_start
        perf_totals["step"] += step_time

        if step % log_interval == 0:
            elapsed = max(time.time() - t0, 1e-9)
            tok_per_sec = since_log_tokens / elapsed
            ex_per_sec = since_log_examples / max(elapsed, 1e-9)
            avg_seq_len = since_log_tokens / max(since_log_examples, 1)
            avg_batch = since_log_examples / max(since_log_steps, 1)

            cuda_alloc, cuda_reserved = _cuda_mem_strings()
            current_lr = (
                optimizer.param_groups[0].get("lr", 0.0)
                if optimizer.param_groups
                else 0.0
            )
            grad_display = "—" if last_grad_norm is None else f"{last_grad_norm:.3f}"
            clipped_display = "YES" if last_clip_triggered else "NO"
            log_stats(
                log,
                "opt",
                step=step,
                lr=f"{current_lr:.6g}",
                grad_norm=grad_display,
                clipped=f"{clipped_display}({_format_float(grad_clip_norm)})",
                amp_scale=f"{amp_scale_value:.3f}",
                cuda_mem_alloc=cuda_alloc,
                cuda_mem_reserved=cuda_reserved,
            )

            log_stats(
                log,
                "kd-hparams",
                step=step,
                alpha=_format_float(current_kd_alpha),
                T=_format_float(current_kd_temperature),
                scheme=kd_scheme_step,
            )

            kd_tok_avg = kd_tok_meter.avg if kd_tok_meter.n else None
            kd_seq_avg = kd_seq_meter.avg if kd_seq_meter.n else None
            loss_stats = {
                "step": step,
                "ce_seq": _format_float(ce_seq_meter.avg),
                "ce_tok": _format_float(ce_tok_meter.avg),
                "kd_seq": _format_float(kd_seq_avg) if kd_seq_avg is not None else "—",
                "kd_tok": _format_float(kd_tok_avg) if kd_tok_avg is not None else "—",
                "total_seq": _format_float(total_seq_meter.avg),
                "total_tok": _format_float(tok_meter.avg),
                "seq_len": _format_float(avg_seq_len),
                "batch": _format_float(avg_batch),
                "reduction": "mean_over(batch,seq)",
                "base": "e",
            }
            log_stats(log, "loss", **loss_stats)

            entropy_stats = {
                "step": step,
                "H_student": _format_float(entropy_student_meter.avg),
                "H_teacher": _format_float(entropy_teacher_meter.avg) if entropy_teacher_meter.n else "—",
                "base": "e",
            }
            log_stats(log, "entropy", **entropy_stats)

            counter_stats = {
                "step": step,
                "tokens_seen": tokens_seen_total,
                "examples_seen": examples_seen_total,
                "steps": step,
            }
            log_stats(log, "counter", **counter_stats)

            loss_ema_stats = {
                "step": step,
                "ce_tok": _format_float(ema_ce_tok) if ema_ce_tok is not None else "—",
                "kd_tok": _format_float(ema_kd_tok) if ema_kd_tok is not None else "—",
                "total_tok": _format_float(ema_total_tok) if ema_total_tok is not None else "—",
                "window": f"ema@{ema_window} steps",
            }
            log_stats(log, "loss-ema", **loss_ema_stats)

            pad_frac = pad_tokens_since_log / max(since_log_total_tokens, 1)
            attn_cov = (
                attn_mask_ones_since_log / max(attn_mask_elems_since_log, 1)
                if attn_mask_elems_since_log
                else None
            )
            data_stats = {
                "step": step,
                "pad_frac": _format_float(pad_frac),
                "attn_mask_coverage": _format_float(attn_cov) if attn_cov is not None else "n/a",
            }
            log_stats(log, "data", **data_stats)

            if kd_tok_meter.n:
                log_stats(
                    log,
                    "divergence",
                    step=step,
                    kl_forward=_format_float(kd_tok_meter.avg),
                    kl_reverse=_format_float(divergence_meter.avg),
                    mse=_format_float(logit_delta_meter.avg),
                    base="e",
                    T=_format_float(current_kd_temperature),
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

            if teacher and (last_top1_match is not None or last_top5_match is not None or last_logprob_corr is not None):
                align_stats = {
                    "step": step,
                    "top1": _format_float(last_top1_match) if last_top1_match is not None else "—",
                    "top5": _format_float(last_top5_match) if last_top5_match is not None else "—",
                    "corr(logp_t,logp_s)": _format_float(last_logprob_corr) if last_logprob_corr is not None else "—",
                }
                log_stats(log, "align", **align_stats)

            avg_loader = perf_totals["loader"] / max(since_log_steps, 1)
            avg_teacher = perf_totals["teacher"] / max(since_log_steps, 1)
            avg_student = perf_totals["student"] / max(since_log_steps, 1)
            avg_kd = perf_totals["kd"] / max(since_log_steps, 1)
            avg_step = perf_totals["step"] / max(since_log_steps, 1)
            avg_ckpt = ckpt_time_accum / max(since_log_steps, 1)
            perf_stats = {
                "step": step,
                "t_step": f"{avg_step:.3f}s",
                "t_loader": f"{avg_loader:.3f}s",
                "t_teacher": f"{avg_teacher:.3f}s",
                "t_student": f"{avg_student:.3f}s",
                "t_kd": f"{avg_kd:.3f}s",
                "t_ckpt": f"{avg_ckpt:.3f}s",
                "tok/s": f"{tok_per_sec:.0f}",
                "ex/s": f"{ex_per_sec:.1f}",
            }
            log_stats(log, "perf", **perf_stats)

            if kd_active_step and len(distill_history) == distill_history.maxlen:
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

            tok_meter = Meter()
            batch_meter = Meter()
            ce_tok_meter = Meter()
            ce_seq_meter = Meter()
            kd_tok_meter = Meter()
            kd_seq_meter = Meter()
            total_seq_meter = Meter()
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
            for key in perf_totals:
                perf_totals[key] = 0.0
            ckpt_time_accum = 0.0
            t0 = time.time()

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
                "batches": metrics.get("batches", "n/a"),
                "examples": metrics.get("examples", "n/a"),
                "tokens": metrics.get("tokens", "n/a"),
                "t_eval": f"{eval_time:.2f}s",
                "loss": _format_float(val_loss),
                "ppl": _format_float(metrics["val_ppl"]),
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
                    prefix="best",
                    step=step,
                    size=_format_bytes(best_size),
                    t_save=f"{best_time:.2f}s",
                    prune=f"best(k={best_ckpt_keep_k},ret={best_ckpt_retention_secs})",
                    uploaded="YES" if best_uri else "NO",
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
                    prefix="best-versioned",
                    step=step,
                    size=_format_bytes(vers_size),
                    t_save=f"{vers_time:.2f}s",
                    prune=f"best(k={best_ckpt_keep_k},ret={best_ckpt_retention_secs})",
                    filename=vers_name,
                )

                # Prune older best checkpoints per policy
                prune_t0 = time.time()
                try:
                    _prune_best_ckpts(
                        local_outdir=state["local_outdir"],
                        log=log,
                        keep_k=best_ckpt_keep_k,
                        retention_secs=best_ckpt_retention_secs,
                        now=time.time(),
                    )
                    prune_time = time.time() - prune_t0
                    ckpt_time_accum += prune_time
                    log_stats(
                        log,
                        "ckpt",
                        prefix="best-prune",
                        step=step,
                        prune=f"best(k={best_ckpt_keep_k},ret={best_ckpt_retention_secs})",
                        t_prune=f"{prune_time:.2f}s",
                    )
                except Exception as e:
                    ckpt_time_accum += time.time() - prune_t0
                    log.warning(f"[ckpt] best prune error: {e}")

            model.train()

            # Step-based checkpointing
            if save_every > 0 and step % save_every == 0:
                step_local, step_uri, step_ckpt_time, step_size = _save_ckpt(
                    ckptio, state, step, filename=f"ckpt_step_{step}.pt", log=log
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
                    prefix="step",
                    step=step,
                    size=_format_bytes(step_size),
                    t_save=f"{step_ckpt_time:.2f}s",
                    prune=f"step(k={step_ckpt_keep_k},ret={step_ckpt_retention_secs})",
                    uploaded="YES" if step_uri else "NO",
                )
                # Prune older step checkpoints per policy (age &/or count)
                prune_t0 = time.time()
                try:
                    _prune_step_ckpts(
                        local_outdir=state["local_outdir"],
                        log=log,
                        retention_secs=step_ckpt_retention_secs,
                        keep_k=step_ckpt_keep_k,
                        now=time.time(),
                    )
                    prune_time = time.time() - prune_t0
                    ckpt_time_accum += prune_time
                    log_stats(
                        log,
                        "ckpt",
                        prefix="step-prune",
                        step=step,
                        prune=f"step(k={step_ckpt_keep_k},ret={step_ckpt_retention_secs})",
                        t_prune=f"{prune_time:.2f}s",
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
                    prefix="time",
                    step=step,
                    size=_format_bytes(time_size),
                    t_save=f"{time_elapsed:.2f}s",
                    prune=f"time(k={time_ckpt_keep_k},ret={time_ckpt_retention_secs})",
                    filename=fname,
                )
                last_time_ckpt_ts = now_ts
                state["last_time_ckpt_ts"] = last_time_ckpt_ts  # persist within this run

                # Prune older time-based checkpoints to save disk
                prune_t0 = time.time()
                try:
                    _prune_time_ckpts(
                        local_outdir=state["local_outdir"],
                        log=log,
                        retention_secs=time_ckpt_retention_secs,
                        keep_k=time_ckpt_keep_k,
                        now=now_ts,
                    )
                    prune_time = time.time() - prune_t0
                    ckpt_time_accum += prune_time
                    log_stats(
                        log,
                        "ckpt",
                        prefix="time-prune",
                        step=step,
                        prune=f"time(k={time_ckpt_keep_k},ret={time_ckpt_retention_secs})",
                        t_prune=f"{prune_time:.2f}s",
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