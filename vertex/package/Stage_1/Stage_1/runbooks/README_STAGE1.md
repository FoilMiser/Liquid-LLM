# Stage-1 Vertex AI Runbook

Recommended Vertex AI Custom Training configuration:

- **Worker pool spec**
  - Machine: `g2-standard-8`
  - Accelerator: `NVIDIA_L4` (count=1)
  - Container: `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest`
  - Local package path: `vertex/package/Stage_1`
  - Python module: `Stage_1.cli`

Populate the Vertex console form as follows:

| Field | Value |
| ----- | ----- |
| Python module | `Stage_1.cli` |
| Command line arguments | Paste the block below |
| Output directory | `gs://liquid-llm-bucket-2/stage1/checkpoints/vertex_runs` |

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
--use_grad_ckpt=true
--dtype=bfloat16
--device=cuda
--hf_secret_name=hf_token
```

If you encounter OOMs or throughput is insufficient, upgrade to `a2-highgpu-1g` with an `NVIDIA_A100_40GB` accelerator using the same arguments.
