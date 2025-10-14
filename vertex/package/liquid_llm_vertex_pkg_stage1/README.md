# liquid_llm_vertex_pkg_stage1

Production-ready Vertex AI custom training package for Liquid LLM Stage-1 KD.

## Quick start on Vertex AI (L4)

1. Zip this folder and upload to Cloud Storage or directly reference from Vertex.
2. Create a custom training job with container image `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest`.
3. Set machine type `g2-standard-8` with `NVIDIA_L4` accelerator (1x) and attach the GCS bucket containing the package.
4. Configure the entrypoint to `python -m stage1.cli`.
5. Set environment variable `HF_TOKEN` via Secret Manager binding or job env var for Hugging Face access.
6. Provide arguments as needed; defaults match the Stage-1 pipeline buckets.

## Upgrading to larger GPUs

To run in online teacher mode or with larger per-device batches, simply change the machine configuration to `a2-highgpu-1g` (A100) or `a3-highgpu-1g` (H100). No code changes are required—pass `--teacher-mode=online` for on-the-fly teacher logits.

## FlashAttention wheel

FlashAttention is installed at runtime from `gs://liquid-llm-bucket-2/FlashAttention/flash_attn-2.8.3+cu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl`. The CLI automatically performs the download and install before training.

## Logs and checkpoints

Console logs stream to Vertex automatically. The trainer emits JSON lines with metrics. Checkpoints and metadata are saved locally under `/tmp/vertex_run/` and mirrored to the configured GCS output directory. TensorBoard summaries can be synced by providing `--tb-gcs-uri`.

## Testing locally

```bash
pip install -e .
pytest tests
```

This will validate FlashAttention availability (skipping gracefully when missing), tool-use utilities, and KD loss behaviour.
