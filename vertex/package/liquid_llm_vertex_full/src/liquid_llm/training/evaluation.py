"""Evaluation utilities for student and teacher models."""

from __future__ import annotations

from typing import Dict

import torch

from .metrics import (
    Meter,
    count_active_tokens,
    cross_entropy,
    get_logits,
    perplexity_from_loss,
)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader,
    *,
    device: str = "cuda",
    precision: str = "no",
) -> Dict[str, float]:
    """Run evaluation and return aggregate metrics."""

    was_training = model.training
    model.eval()
    loss_meter = Meter()
    batch_meter = Meter()

    device_type = "cuda" if device.startswith("cuda") else device
    amp_enabled = device_type == "cuda" and precision in {"fp16", "bf16"}
    amp_dtype = None
    if amp_enabled:
        amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    token_count = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type, enabled=amp_enabled, dtype=amp_dtype):
            logits = get_logits(model(input_ids))
            loss = cross_entropy(logits, labels)

        tokens = count_active_tokens(labels, input_ids.numel())
        loss_meter.update(loss.item(), k=tokens)
        token_count += tokens
        batch_meter.update(loss.item(), k=input_ids.size(0))

    if was_training:
        model.train()
    else:
        model.eval()

    avg_loss = loss_meter.avg
    metrics = {
        "val_loss": avg_loss,
        "val_ppl": perplexity_from_loss(avg_loss),
        "val_loss_per_batch": batch_meter.avg,
        "val_tokens": token_count,
    }
    return metrics
