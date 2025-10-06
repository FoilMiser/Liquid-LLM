import torch
from .metrics import cross_entropy, Meter

@torch.no_grad()
def evaluate(model, val_loader, device='cuda'):
    model.eval()
    meter = Meter()
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        logits = model(input_ids)
        loss = cross_entropy(logits, labels)
        meter.update(loss.item(), k=input_ids.size(0))
    model.train()
    return {'val_loss': meter.avg, 'val_ppl': float(torch.exp(torch.tensor(meter.avg)))}
