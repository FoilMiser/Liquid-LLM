# Stage-1 Vertex AI Runbook

Recommended Vertex AI Custom Training configuration:

- **Worker pool spec**
  - Machine: `g2-standard-8`
  - Accelerator: `NVIDIA_L4` (count=1)
  - Container: `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest`
  - Local package path: `vertex/package/Stage_1`
  - Python module: `Stage_1.launcher`

Populate the Vertex console form as follows:

| Field | Value |
| ----- | ----- |
| Python module | `Stage_1.launcher` |
| Command line arguments | Paste the block below |
| Output directory | `gs://liquid-llm-bucket-2/stage1/checkpoints/vertex_runs` |

The CLI validates the resume checkpoint, teacher model selection, and dataset
manifest before launching training. It also generates a UTC `run_id` and
appends it to the output path automatically, so no shell `$(date ...)` logic is
necessary.

```
--resume_gcs_uri=gs://liquid-llm-bucket-2/stage1/stage1.pt
--output_gcs_uri=gs://liquid-llm-bucket-2/stage1/checkpoints/vertex_runs
--teacher_name=meta-llama/Meta-Llama-3.1-8B
--dataset_cfg=gs://liquid-llm-bucket-2/datasets/stage1.jsonl
--seq_len=1024
--block_size=1024
--train_steps=250000
--batch_size=8
--throughput_tokens=32768
--use_flash_attn=true
--fa_wheel_gcs_uri=gs://YOUR_BUCKET/wheels/flash_attn-2.5.8-cp310-cp310-manylinux2014_x86_64.whl
--use_grad_ckpt=true
--dtype=bfloat16
--device=cuda
--hf_secret_name=hf_token
```

## FlashAttention wheel handling

The `Stage_1.launcher` module consumes the FlashAttention wheel flags before
delegating to the training CLI. You can provide either
`--fa_wheel_gcs_uri=gs://...` or `--fa_wheel_url=https://...` when launching the
job. The launcher will attempt to download and install the wheel and strip those
flags from the arguments passed to `Stage_1.cli`, preventing "unrecognized
arguments" errors.

Environment variables offer the same functionality when CLI flags are
inconvenient:

- `FA_WHEEL_GCS_URI`
- `FA_WHEEL_URL`
- `STAGE1_USE_FLASH_ATTN`

The `STAGE1_USE_FLASH_ATTN` variable mirrors the `--use_flash_attn` flag and is
automatically set by the launcher, ensuring the trainer maintains the requested
attention backend even if the CLI flag is omitted.

If you encounter OOMs or throughput is insufficient, upgrade to `a2-highgpu-1g`
with an `NVIDIA_A100_40GB` accelerator using the same arguments.
