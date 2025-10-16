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

## Dry run sanity check

The package supports a lightweight dry run to validate data bootstrap, checkpointing, and evaluation without requiring a GPU. Run the following command on a CPU VM or your workstation:

```bash
python -m stage1.cli \
  --dry-run=true \
  --limit-batches=2 \
  --metrics-interval=1 \
  --prepare-data=skip \
  --dataset-manifest=/path/to/local/manifest.jsonl \
  --teacher-mode=precompute \
  --teacher-logits-dir=/path/to/logits
```

The dry run builds two batches, performs a single optimizer step, runs one evaluation pass, and exits cleanly after saving `last.pt` plus emitting a few metric lines.

## Knowledge distillation modes

* **`--teacher-mode=online`** loads the Hugging Face teacher model at startup. The CLI prints the teacher vocab size and keeps the teacher under `torch.inference_mode()` for every batch. Use this when a GPU and HF token are available; KD losses are computed on the fly.
* **`--teacher-mode=precompute`** expects `.pt` or `.npy` logits per sample in `--teacher-logits-dir`. The data loader validates shapes (`[seq, vocab]`), pads/truncates when necessary, and surfaces a `teacher_status` flag per batch. If >10% of batches in an epoch are missing logits, the trainer logs a prominent warning.

## Metrics and observability

Training metrics stream to `<run_dir>/metrics.jsonl` and are mirrored to GCS after every append. Each record includes:

* `global_step`, `train_loss`, `ce_loss`, `kd_loss`, `logit_l2`
* `lr`, `tokens_per_sec`, `examples_per_sec`, `grad_norm`, `gpu_mem_alloc_MB`
* `val_ppl` when validation runs

During the first ~2k steps on an L4 you should expect `tokens_per_sec` to stabilise after the warmup and `grad_norm` to stay finite (typically <10). Validation perplexity is logged whenever `--eval-every` divides the current step.

The startup banner also captures seed, precision, SDPA status, FlashAttention wheel installation status, and data bootstrap summaries including per-dataset shard counts and auto-generated sample IDs.

## Reliability and recovery

The trainer guards against common failure modes:

* **NaN/Inf detection** – the first non-finite loss or gradient skips the update and logs a warning; a consecutive occurrence triggers a `crash_dump.pt` containing model/optimizer state and batch shapes.
* **AMP overflow** – if `GradScaler` detects overflow, the optimizer step is skipped, the scale is backed off, and the same batches are retried.
* **CUDA OOM** – the current `grad_accum_steps` is halved (down to 1) and the offending step is retried. A second OOM writes a crash dump and raises with a clear message.
* **Signals** – SIGTERM/SIGINT triggers a graceful shutdown that saves `last.pt`, flushes `metrics.jsonl`, uploads both, and exits with status 0.

Crash dumps, checkpoints (`best.pt`, `last.pt`), `run_meta.json`, and `frozen_mask.json` are uploaded immediately after creation with a post-upload integrity check (`gcloud storage ls`).

## Resuming training

To resume from the latest checkpoint, point `--resume_gcs_uri` at the desired artifact (typically `last.pt`). The CLI also writes `args_snapshot.json` with the original argument set and provenance metadata (`HF_TOKEN_PRESENT`, `GOOGLE_CLOUD_PROJECT`, git commit) to aid reproducibility.

## Logs and checkpoints

Console logs stream to Vertex automatically. The trainer emits JSON lines with metrics. Checkpoints and metadata are saved locally under `/tmp/vertex_run/` and mirrored to the configured GCS output directory. TensorBoard summaries can be synced by providing `--tb-gcs-uri`.

## Testing locally

```bash
pip install -e .
pytest tests
```

This will validate FlashAttention availability (skipping gracefully when missing), tool-use utilities, and KD loss behaviour.
