import torch
import torch.nn as nn

class GELU(nn.Module):
    def forward(self, x): return torch.nn.functional.gelu(x)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4*d_model), GELU(), nn.Linear(4*d_model, d_model), nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class StudentLM(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=10, n_heads=12, dropout=0.0, pad_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(4096, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.register_buffer("pos_ids", torch.arange(0, 4096).unsqueeze(0), persistent=False)

    def forward(self, input_ids):
        b, t = input_ids.size()
        pos = self.pos_ids[:, :t]
        x = self.embed(input_ids) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

def build_student_model(vocab_size: int, pad_id: int, d_model: int, n_layers: int, n_heads: int, dropout: float = 0.0):
    return StudentLM(vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout, pad_id=pad_id)
