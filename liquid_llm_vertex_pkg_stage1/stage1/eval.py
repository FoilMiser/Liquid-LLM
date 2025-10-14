"""Evaluation helpers."""

import logging
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


def evaluate_perplexity(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(outputs.loss.item())
    model.train()
    if not losses:
        return float("inf")
    mean_loss = sum(losses) / len(losses)
    return float(torch.exp(torch.tensor(mean_loss)))


def evaluate_tool_accuracy(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids).logits
            preds = logits.argmax(dim=-1)
            match = (preds == labels).float()
            correct += match.mean().item()
            total += 1
    model.train()
    if total == 0:
        return 0.0
    return correct / total


def run_eval(model: torch.nn.Module, datamodule, device: torch.device) -> Dict[str, float]:
    ppl = evaluate_perplexity(model, datamodule.val_dataloader(), device)
    acc = evaluate_tool_accuracy(model, datamodule.val_dataloader(), device)
    metrics = {"perplexity": ppl, "tool_accuracy": acc}
    LOGGER.info("Eval metrics: %s", metrics)
    return metrics


__all__ = ["run_eval", "evaluate_perplexity", "evaluate_tool_accuracy"]
