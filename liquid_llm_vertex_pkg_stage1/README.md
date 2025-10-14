# liquid_llm_vertex_pkg_stage1

This package contains the Stage 1 knowledge distillation training pipeline for Liquid LLM on Google Vertex AI. It supports launching custom training jobs that fine-tune a widened student model from `gs://liquid-llm-bucket-2/stage1/stage1.pt` using teacher guidance from `meta-llama/Llama-3.2-3B`, supervised tool traces, and Vertex-friendly logging and checkpointing.

## Vertex AI Custom Job Configuration

* **Container image:** `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest`
* **Machine type:** `g2-standard-8`
* **Accelerators:** `1x NVIDIA_L4`
* **Python package URI:** Point to a GCS location containing the built source distribution of this repository.
* **Python module:** `stage1.cli`

### Arguments

Paste the following block into the Vertex console **Arguments** field:

```
--mode=train
--resume_gcs_uri=gs://liquid-llm-bucket-2/stage1/stage1.pt
--output_gcs_uri=gs://liquid-llm-bucket-2/stage1/Checkpoints/vertex-runs
--teacher_id=meta-llama/Llama-3.2-3B
--teacher_mode=precompute
--dataset_manifest=gs://liquid-llm-bucket-2/datasets/stage1/manifests/stage1.jsonl
--seq_len=1024
--block_size=1024
--precision=bfloat16
--lr=2.5e-4 --weight_decay=0.1 --betas=0.9,0.95
--warmup_steps=3000
--max_steps=120000
--eval_every=1000
--save_every=2000
--tool_use_ratio=0.08
--kd_temperature=2.0
--kd_alpha_start=0.7 --kd_alpha_end=0.4 --kd_anneal_pct=0.3
--keep_old_logit_l2=0.1 --keep_old_logit_l2_fade_step=30000
--fa_wheel_gcs=gs://liquid-llm-bucket-2/FlashAttention/flash_attn-2.8.3+cu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

## Package Layout

The `stage1` package contains modular components for runtime setup, model and teacher loading, data ingest, training, evaluation, tool simulation, and utility helpers. Tests under `tests/` provide lightweight verification of the attention backend selection logic, loss numerics, and tool-use pipeline.
