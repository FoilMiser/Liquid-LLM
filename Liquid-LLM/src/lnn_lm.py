
# src/lnn_lm.py
import torch
import torch.nn as nn
from .ltc_cell import LTCCell

class LiquidLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.liquid = nn.ModuleList([LTCCell(d_model, d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.hidden_size = d_model
        self.n_layers = n_layers

    def init_state(self, batch_size, device):
        return [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.n_layers)]

    def forward(self, input_ids, state=None):
        B, T = input_ids.size()
        x = self.embed(input_ids)
        if state is None:
            state = self.init_state(B, x.device)
        logits = []
        for t in range(T):
            h = x[:, t, :]
            for i, cell in enumerate(self.liquid):
                s = state[i]
                s = cell(s, h)
                h = h + s  # residual
                state[i] = s
            h = self.norm(h)
            logits.append(self.lm_head(h))
        return torch.stack(logits, dim=1), state
