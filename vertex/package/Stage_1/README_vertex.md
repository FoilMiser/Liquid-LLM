# Vertex AI Stage-1 Training Job Setup

Use the following configuration when launching a Vertex AI custom training job for Stage-1.

## Vertex AI custom job settings

- **Python module**: `Stage_1.vertex.entrypoint`
- **Container image**: `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest`
- **Environment variable**: `PYTHONPATH=/root/.local/lib/python3.10/site-packages:${PYTHONPATH}`

### Arguments (space-separated)

```
--resume_gcs_uri=gs://liquid-llm-bucket-2/stage1/stage1.pt \
--output_gcs_uri=gs://liquid-llm-bucket-2/stage1/checkpoints/vertex_runs \
--teacher_name=meta-llama/Meta-Llama-3.1-8B \
--dataset_cfg=gs://liquid-llm-bucket-2/datasets/stage1.jsonl \
--seq_len=1024 \
--block_size=1024 \
--train_steps=250000 \
--batch_size=8 \
--throughput_tokens=32768 \
--use_flash_attn=true \
--fa_wheel_gcs_uri=gs://liquid-llm-bucket-2/FlashAttention/flash_attn-2.8.3+cu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl \
--use_grad_ckpt=true \
--dtype=bfloat16 \
--device=cuda \
--hf_secret_name=hf_token
```

## Why this configuration matters

Vertex installs uploaded Python packages into `/root/.local/lib/python3.10/site-packages`, which is not on `sys.path` by default. Setting `PYTHONPATH` to include that directory ensures Python can discover `Stage_1.vertex.entrypoint`. The entrypoint now also adds the user site-packages directory programmatically, so the module remains importable even if the environment variable is omitted.

## Optional verification command

Run this snippet inside the Vertex container to verify the package is discoverable:

```
python3 -c "import sys,site,importlib; print(site.getusersitepackages() in sys.path); importlib.import_module('Stage_1.vertex.entrypoint'); print('entrypoint import OK')"
```
