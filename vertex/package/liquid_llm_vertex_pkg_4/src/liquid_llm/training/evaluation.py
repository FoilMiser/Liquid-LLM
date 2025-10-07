import torch

from .metrics import Meter, cross_entropy


def _extract_logits(output):
    """Return tensor logits from either an HF output struct or raw tensor."""
    if hasattr(output, "logits"):
        return output.logits
    return output


@torch.no_grad()
def evaluate(model, val_loader, device: str = "cuda"):
    was_training = model.training
    model.eval()
    meter = Meter()
    total_tokens = 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        output = model(input_ids)
        logits = _extract_logits(output)
        loss = cross_entropy(logits, labels)
        contrib_tokens = int((labels != -100).sum().item())
        if contrib_tokens == 0:
            contrib_tokens = labels.numel()
        total_tokens += contrib_tokens
        meter.update(loss.item(), k=contrib_tokens)
    if was_training:
        model.train()
    avg_loss = meter.avg if meter.n else 0.0
    ppl = float(torch.exp(torch.tensor(avg_loss))) if total_tokens else float("inf")
    return {"val_loss": avg_loss, "val_ppl": ppl, "tokens": total_tokens}
