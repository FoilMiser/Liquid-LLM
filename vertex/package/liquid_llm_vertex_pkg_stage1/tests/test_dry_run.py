import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1.train import Trainer
from stage1.utils import AnnealingSchedule


class _MiniModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 16) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 8)
        self.ln = torch.nn.LayerNorm(8)
        self.lm_head = torch.nn.Linear(8, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(input_ids)
        hidden = self.ln(hidden)
        return self.lm_head(hidden)


def _collate(batch):
    input_ids = torch.stack([item[0] for item in batch])
    return {"input_ids": input_ids}


def test_trainer_dry_run(tmp_path):
    torch.manual_seed(0)
    dataset = TensorDataset(torch.randint(0, 10, (4, 6)))
    train_loader = DataLoader(dataset, batch_size=2, collate_fn=_collate)
    val_loader = DataLoader(dataset, batch_size=2, collate_fn=_collate)
    model = _MiniModel()
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        output_gcs_uri=None,
        run_id="dry",
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        warmup_steps=0,
        max_steps=5,
        kd_temperature=2.0,
        kd_alpha_schedule=AnnealingSchedule(0.7, 0.4, 0.3),
        ce_beta_schedule=AnnealingSchedule(0.3, 0.6, 0.3),
        logit_l2_gamma_schedule=AnnealingSchedule(0.0, 0.0, 1.0),
        logit_reference=None,
        precision="fp32",
        teacher=None,
        teacher_mode="precompute",
        teacher_logits_dir=None,
        eval_every=1,
        save_every=1,
        grad_accum_steps=1,
        metrics_interval=1,
        dry_run=True,
    )
    trainer.train()
    assert trainer.global_step == 1
    metrics_path = Path(tmp_path) / "metrics.jsonl"
    assert metrics_path.exists()
    last_ckpt = Path(tmp_path) / "last.pt"
    assert last_ckpt.exists()
