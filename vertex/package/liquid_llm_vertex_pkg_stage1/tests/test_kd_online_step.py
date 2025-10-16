import sys
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1.train import Trainer
from stage1.utils import AnnealingSchedule


class _ToyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 8, seq_len: int = 4) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 16)
        self.ln = torch.nn.LayerNorm(16)
        self.lm_head = torch.nn.Linear(16, vocab_size)
        self.seq_len = seq_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(input_ids)
        hidden = self.ln(hidden)
        return self.lm_head(hidden)


class _Teacher:
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq = input_ids.shape
        vocab = 8
        torch.manual_seed(0)
        return torch.randn(batch, seq, vocab)


def test_trainer_computes_online_kd(tmp_path):
    torch.manual_seed(0)
    dataset = TensorDataset(torch.randint(0, 8, (2, 5), dtype=torch.long))

    def collate(batch):
        input_ids = torch.stack([item[0] for item in batch])
        return {"input_ids": input_ids}

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate)
    model = _ToyModel()
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=None,
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        output_gcs_uri=None,
        run_id="test",
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        warmup_steps=0,
        max_steps=1,
        kd_temperature=2.0,
        kd_alpha_schedule=AnnealingSchedule(0.7, 0.4, 0.3),
        ce_beta_schedule=AnnealingSchedule(0.3, 0.6, 0.3),
        logit_l2_gamma_schedule=AnnealingSchedule(0.0, 0.0, 1.0),
        logit_reference=None,
        precision="fp32",
        teacher=_Teacher(),
        teacher_mode="online",
        teacher_logits_dir=None,
        eval_every=0,
        save_every=0,
        metrics_interval=1,
    )
    trainer.train()
    metrics = trainer._last_metrics
    assert metrics is not None
    assert math.isfinite(metrics["kd_loss"])
    assert metrics["kd_loss"] > 0
