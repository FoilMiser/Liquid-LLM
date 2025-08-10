
# src/ltc_cell.py
import torch
import torch.nn as nn
from torchdiffeq import odeint

class LTCFunc(nn.Module):
    def __init__(self, hidden, input_size):
        super().__init__()
        self.win = nn.Linear(input_size, hidden, bias=True)
        self.wrec = nn.Linear(hidden, hidden, bias=False)
        self.tau = nn.Parameter(torch.ones(hidden))
        self.act = nn.SiLU()

    def forward(self, t, x, u):
        return ((-x + self.act(self.win(u) + self.wrec(x))) / (self.tau + 1e-3))

class LTCCell(nn.Module):
    def __init__(self, input_size, hidden, steps=2):
        super().__init__()
        self.func = LTCFunc(hidden, input_size)
        self.steps = steps
        self.hidden = hidden

    def forward(self, x_prev, u_t):
        t = torch.tensor([0.0, 1.0], device=u_t.device)
        def f(t_scalar, x_state):
            return self.func(t_scalar, x_state, u_t)
        x = odeint(f, x_prev, t, method='rk4')[-1]
        return x
