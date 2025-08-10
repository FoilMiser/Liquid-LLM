
# src/train_kd.py
import os, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator
from transformers import PreTrainedTokenizerFast
from .lnn_lm import LiquidLM

def kd_loss(student_logits, labels, teacher_probs, alpha=0.8):
    student = student_logits[:, :-1, :]
    tprobs  = teacher_probs[:, :-1, :].clamp_min(1e-8)
    labels  = labels[:, 1:]
    ce = F.cross_entropy(student.reshape(-1, student.size(-1)), labels.reshape(-1))
    logp_s = F.log_softmax(student, dim=-1)
    kl = F.kl_div(logp_s, tprobs, reduction='batchmean')
    return alpha * kl + (1 - alpha) * ce

def collate_batch(batch, tok, ctx=256):
    texts = [ex['text'] for ex in batch if ex.get('text')]
    toks = tok(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=ctx)
    input_ids = toks['input_ids']
    labels = input_ids.clone()
    return {'input_ids': input_ids, 'labels': labels}

def train_small_kd(tokenizer_path, dataset_name='wikitext', subset='wikitext-2-raw-v1',
                   steps=50, lr=3e-4, d_model=256, n_layers=2, vocab_size=32000, batch_size=2):
    accelerator = Accelerator()
    device = accelerator.device
    tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    model = LiquidLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers).to(device)

    ds = load_dataset(dataset_name, subset, split='train')
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    collate_fn=lambda b: collate_batch(b, tok))

    model, dl = accelerator.prepare(model, dl)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for step, batch in enumerate(dl):
        if step >= steps: break
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        with accelerator.autocast():
            logits, _ = model(input_ids)
            vocab = logits.size(-1)
            teacher_probs = torch.full_like(logits, 1.0 / vocab)  # placeholder
            loss = kd_loss(logits, labels, teacher_probs, alpha=0.8)
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if accelerator.is_main_process and (step+1) % 10 == 0:
            print(f"step {step+1}: loss={loss.item():.4f}")
    if accelerator.is_main_process:
        os.makedirs("models/student", exist_ok=True)
        torch.save(model.state_dict(), "models/student/liquidlm_smoke.pt")
        print("Saved models/student/liquidlm_smoke.pt")
