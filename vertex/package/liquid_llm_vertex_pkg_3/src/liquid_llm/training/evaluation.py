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
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        output = model(input_ids)
        logits = _extract_logits(output)
        loss = cross_entropy(logits, labels)
        meter.update(loss.item(), k=input_ids.size(0))
    if was_training:
        model.train()
    return {"val_loss": meter.avg, "val_ppl": float(torch.exp(torch.tensor(meter.avg)))}
