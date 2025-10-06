from dataclasses import dataclass
import torch

@dataclass
class Meter:
    total: float = 0.0
    n: int = 0
    def update(self, x: float, k: int = 1):
        self.total += float(x) * k
        self.n += k
    @property
    def avg(self):
        return self.total / max(self.n, 1)

def cross_entropy(logits, labels):
    # shift-less CE: token-wise CE
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
    )
