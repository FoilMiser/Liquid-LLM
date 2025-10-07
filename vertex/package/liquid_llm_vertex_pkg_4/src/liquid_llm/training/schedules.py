from torch.optim.lr_scheduler import LambdaLR

def build_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine decay to 10% of lr
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        import math
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return LambdaLR(optimizer, lr_lambda)
