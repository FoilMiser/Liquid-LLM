import math
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer, warmup_steps, total_steps, peak_scale: float = 1.0, min_scale: float = 0.1):
    peak_scale = float(peak_scale) if peak_scale is not None else 1.0
    min_scale = float(min_scale)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return peak_scale * float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return peak_scale * (min_scale + (1.0 - min_scale) * cosine)

    return LambdaLR(optimizer, lr_lambda)
