"""Evaluation helpers for validation perplexity and tool metrics."""
from __future__ import annotations

import math
from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader

from .losses import ce_loss
from .utils import configure_logging

logger = configure_logging()


def evaluate_perplexity(model: torch.nn.Module, dataloader: DataLoader) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.lm_head.weight.device)
            logits = model(input_ids)
            loss = ce_loss(logits[:, :-1], input_ids[:, 1:])
            losses.append(loss.item())
    model.train()
    if not losses:
        return float("inf")
    return math.exp(sum(losses) / len(losses))


def compute_tool_accuracy(preds: Iterable[str], targets: Iterable[str]) -> float:
    total = 0
    correct = 0
    for pred, target in zip(preds, targets):
        total += 1
        if pred.strip() == target.strip():
            correct += 1
    return correct / total if total else 0.0
