import sys
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1.train import Trainer
from stage1.utils import AnnealingSchedule


class _TinyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 4)
        self.ln = torch.nn.LayerNorm(4)
        self.lm_head = torch.nn.Linear(4, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(input_ids)
        hidden = self.ln(hidden)
        return self.lm_head(hidden)


def test_checkpoint_uploads_immediately(tmp_path, monkeypatch):
    dataset = TensorDataset(torch.randint(0, 8, (1, 5)))

    def collate(batch):
        return {"input_ids": torch.stack([item[0] for item in batch])}

    loader = DataLoader(dataset, batch_size=1, collate_fn=collate)
    model = _TinyModel()
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=None,
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        output_gcs_uri="gs://bucket/path",
        run_id="run123",
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
        teacher=None,
        teacher_mode="precompute",
        teacher_logits_dir=None,
        eval_every=0,
        save_every=0,
    )
    monkeypatch.setenv("STAGE1_DATA_PROVENANCE_DIR", str(tmp_path))
    with mock.patch("stage1.train.local_to_gcs") as upload_mock:
        trainer._save_checkpoint("last.pt", 0, 1.0, {"loss_total": 0.0})
        destinations = [call.args[1] for call in upload_mock.call_args_list]
        assert f"gs://bucket/path/run123/last.pt" in destinations
        assert f"gs://bucket/path/run123/run_meta.json" in destinations
