# Liquid LLM Vertex Package 4 (Annealing)

This package extends the Stage 0 Vertex trainer with first-class support for
knowledge-distillation annealing runs that start at a shorter context window
and then resume at a longer window.

## New command-line flags

The launcher (`python -m trainer.entrypoint --help`) now exposes the following
arguments in addition to the base package:

| Flag | Description |
| ---- | ----------- |
| `--alpha` | Override the initial KD alpha value. Falls back to checkpoint metadata or `0.5`. |
| `--T` | Override the initial KD temperature. Falls back to checkpoint metadata or `1.0`. |
| `--alpha_schedule` | Comma separated `step:value` list for alpha updates (e.g. `0:0.7,400:0.5`). |
| `--temp_schedule` | KD temperature schedule with the same syntax as `--alpha_schedule`. |
| `--eval_ctx_lens` | Comma separated context lengths to evaluate at every evaluation step. |
| `--reset_optim_on_ctx_change` | When resuming with a different `--block_size`, rebuild the optimizer and scheduler (weights are kept). |
| `--grad_accum` | Number of micro batches to accumulate before each optimizer step (keeps tokens/step roughly constant when the context changes). |
| `--lr_peak` | Optional peak learning rate reached at the end of warmup when reinitialising a scheduler. |
| `--rope_scale` | Optional RoPE scaling factor (stored in checkpoint metadata for rotary-enabled models). |

Schedules are parsed eagerly into sorted lists and applied before every train
step. Any CLI value overrides checkpoint metadata; otherwise, resume metadata is
used.

## Multi-context evaluation

When `--eval_ctx_lens` is provided, the trainer keeps the main training
DataLoader untouched and builds additional **validation-only** DataLoaders for
each requested context length. Evaluation metrics are logged per context length
with keys such as `val_loss_student@512` and `val_loss_student@1024` while the
existing single-context log entries remain for backwards compatibility.

## Context length changes & optimizer resets

Checkpoint metadata records the context length used when it was saved. On resume
with a different `--block_size`:

1. The dataloaders are rebuilt for the new context length.
2. Absolute positional embeddings are resized (via interpolation) to cover the
   requested sequence length.
3. If `--reset_optim_on_ctx_change` is set, both optimizer and scheduler are
   reinitialised and the step counter is reset to `0` so a fresh warmup can be
   applied (optionally with `--lr_peak`).

## Checkpoint metadata

Every checkpoint (`best.pt`, step/time snapshots, and `final.pt`) now carries a
small JSON sidecar (`*.meta.json`) and in-file metadata that captures:

* `block_size`, `kd_alpha`, `kd_temperature`
* KD schedules (original strings) and the pending schedule state
* `grad_accum`, `warmup_steps`, `lr`, `lr_peak`
* Positional embedding configuration and any provided RoPE scale
* The evaluation context lengths used for logging

Metadata is logged during save and on resume so downstream jobs can be safely
orchestrated.

## Example runs

**A. 512-token anneal pass**

```
python -m trainer.entrypoint \
  --resume_gcs_uri=gs://liquid-llm-bucket-2/stage0/checkpoints/vertex_runs/20251007-174419/best.pt \
  --block_size=512 \
  --teacher_name=gpt2-xl \
  --dataset_name=wikitext \
  --dataset_config=wikitext-103-raw-v1 \
  --alpha=0.7 \
  --T=1.2 \
  --alpha_schedule=0:0.7,400:0.5,800:0.3 \
  --temp_schedule=0:1.2,800:1.0 \
  --warmup_steps=200 \
  --lr_peak=2.7e-4 \
  --eval_ctx_lens=512,1024 \
  --grad_accum=1 \
  --train_steps=1200 \
  --output_gcs_uri=gs://liquid-llm-bucket-2/stage0/checkpoints/vertex_runs/$(date +%Y%m%d-%H%M%S)
```

**B. 1024-token continuation (resume from the best 512 checkpoint)**

```
python -m trainer.entrypoint \
  --resume_gcs_uri=gs://liquid-llm-bucket-2/stage0/checkpoints/<BEST_FROM_512_ANNEAL>.pt \
  --block_size=1024 \
  --teacher_name=gpt2-xl \
  --dataset_name=wikitext \
  --dataset_config=wikitext-103-raw-v1 \
  --alpha=0.5 \
  --T=1.0 \
  --alpha_schedule=0:0.5,2000:0.4,4000:0.3 \
  --warmup_steps=400 \
  --lr_peak=2.7e-4 \
  --grad_accum=2 \
  --reset_optim_on_ctx_change \
  --eval_ctx_lens=512,1024 \
  --train_steps=6000 \
  --output_gcs_uri=gs://liquid-llm-bucket-2/stage0/checkpoints/vertex_runs/$(date +%Y%m%d-%H%M%S)
```

## Quick smoke checks

* `python -m trainer.entrypoint --help` – verify new CLI arguments appear.
* `python -m trainer.entrypoint --resume_gcs_uri <URI> --block_size=512 --teacher_name=gpt2-xl --dataset_name=wikitext --dataset_config=wikitext-103-raw-v1 --alpha=0.6 --T=1.1 --alpha_schedule=0:0.6,1:0.5 --temp_schedule=0:1.1,1:1.0 --train_steps=0 --eval_every=1 --save_every=1 --log_interval=1 --local_workdir=/tmp/liquid_test --output_gcs_uri=gs://dummy`
  – loads the checkpoint, merges metadata, and exits immediately (no training steps run).
* `python -m trainer.entrypoint --block_size=512 --teacher_name=gpt2-xl --dataset_name=wikitext --dataset_config=wikitext-103-raw-v1 --train_steps=1 --eval_every=1 --save_every=1 --log_interval=1 --alpha=0.5 --T=1.0 --alpha_schedule=0:0.5,1:0.4 --temp_schedule=0:1.0,1:0.9 --eval_ctx_lens=512,1024 --local_workdir=/tmp/liquid_test`
  – executes a single optimizer step to confirm schedule application, multi-context evaluation, and checkpoint metadata logging.
