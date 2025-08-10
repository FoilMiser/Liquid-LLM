
# scripts/smoke_test.py
import torch
from src.lnn_lm import LiquidLM

def main():
    vocab = 32000
    model = LiquidLM(vocab_size=vocab, d_model=256, n_layers=2)
    x = torch.randint(0, vocab, (2, 16))
    logits, state = model(x)
    print("OK - logits shape:", logits.shape)

if __name__ == "__main__":
    main()
