# Stage-1 Vertex Training Package

<!--
Worker pool spec:
local-package-path=vertex/package/Stage_1
python-module=Stage_1.vertex.entrypoint
container-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest
-->

This package bundles the Stage-1 distillation training entrypoint for Vertex AI
Custom Training jobs. Use the module launcher directly to avoid PATH issues:

```bash
python -m Stage_1.vertex.entrypoint --help
```

Vertex's bootstrap command should install the local package into the interpreter
site-packages directory:

```bash
python -m pip install --no-cache-dir --no-user --upgrade .
```

## Vertex configuration notes

- **Worker pool spec**
  - Local package path: `vertex/package/Stage_1`
  - Python module: `Stage_1.vertex.entrypoint`
  - Container image: `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest`
- Do **not** add any `pip install` commands to the Vertex "Arguments" field. The
  package already declares its dependencies.

### Recommended argument block

Use the following newline-separated list in the Vertex console "Arguments"
textbox:

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
--fa_wheel_gcs_uri=gs://liquid-llm-bucket-2/FlashAttention/flash_attn-2.8.3+cu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
--use_grad_ckpt=true
--dtype=bfloat16
--device=cuda
--hf_secret_name=hf_token
```

The CLI validates checkpoints, datasets, and environment compatibility before
launching training, and it logs resolved dependency versions at startup.
