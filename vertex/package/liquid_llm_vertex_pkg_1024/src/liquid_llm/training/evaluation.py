import torch
from contextlib import nullcontext

from .metrics import Meter, cross_entropy


def _extract_logits(output):
    """Return tensor logits from either an HF output struct or raw tensor."""
    if hasattr(output, "logits"):
        return output.logits
    return output


@torch.no_grad()
def evaluate(model, val_loader, device: str = "cuda", autocast_dtype=None):
    was_training = model.training
    model.eval()
    meter = Meter()
    total_tokens = 0
    total_examples = 0
    total_batches = 0
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    use_autocast = False
    if isinstance(autocast_dtype, torch.dtype):
        if device_type == "cuda":
            use_autocast = True
        elif device_type == "cpu" and autocast_dtype == torch.bfloat16:
            use_autocast = True
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        autocast_cm = (
            torch.autocast(device_type=device_type, dtype=autocast_dtype)
            if use_autocast
            else nullcontext()
        )
        with autocast_cm:
            output = model(input_ids)
            logits = _extract_logits(output)
        logits = logits.float()
        loss = cross_entropy(logits, labels)
        contrib_tokens = int((labels != -100).sum().item())
        if contrib_tokens == 0:
            contrib_tokens = labels.numel()
        total_tokens += contrib_tokens
        total_examples += input_ids.size(0)
        total_batches += 1
        meter.update(loss.item(), k=contrib_tokens)
    if was_training:
        model.train()
    avg_loss = meter.avg if meter.n else 0.0
    ppl = float(torch.exp(torch.tensor(avg_loss))) if total_tokens else float("inf")
    return {
        "val_ce": avg_loss,
        "val_ppl": ppl,
        "val_loss": avg_loss,
        "tokens": total_tokens,
        "examples": total_examples,
        "batches": total_batches,
    }
