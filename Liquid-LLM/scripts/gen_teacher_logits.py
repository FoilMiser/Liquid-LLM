
# scripts/gen_teacher_logits.py
"""
Stub to generate/capture teacher probabilities from DeepSeek-R1-0528.
Replace the 'get_teacher_probs' body with your actual teacher call (API or local).
Cache shape: [B, T, V]. Save as .pt for KD.
"""
import torch, os
from pathlib import Path

def get_teacher_probs(batch_texts, tokenizer, vocab_size, temperature=1.5):
    # TODO: Replace with actual teacher inference returning per-token probabilities
    max_len = max(len(t) for t in batch_texts)
    return torch.full((len(batch_texts), max_len, vocab_size), 1.0/vocab_size)

def main():
    os.makedirs("data/teacher_cache", exist_ok=True)
    # Implement your batching over a dataset and save .pt tensors alongside input_ids
    pass

if __name__ == "__main__":
    main()
