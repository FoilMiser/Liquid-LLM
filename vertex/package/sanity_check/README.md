# Vertex Stage-0 Checkpoint Sanity Check

This package runs a compact sanity and smoke test suite for distilled Stage-0 language model checkpoints hosted on Google Cloud Storage. It is intended to run inside Vertex AI Custom Training jobs on a single NVIDIA L4 GPU to validate checkpoints before promotion to production workflows.

## Features

* Deterministic environment and GPU checks
* Robust checkpoint download and integrity validation
* Model reconstruction from stored config or inferred shapes
* Tokenizer round-trip and sample batch encoding tests
* Forward and backward smoke tests with gradient validation
* Lightweight throughput probe and CUDA memory statistics
* Structured JSON logging compatible with Vertex AI log ingestion

## Installation

```
pip install .
```

## Usage

Run locally (CPU) with a local checkpoint path for quick verification:

```
python -m sanity_check.cli \
  --checkpoint_gcs_uri=/path/to/checkpoint.pt \
  --device=cpu \
  --dtype=float32
```

### Vertex AI Launch

The package is ready for single-worker Vertex AI Custom Training jobs using the PyTorch GPU 2.4 Python 3.10 container. Example command:

```
gcloud ai custom-jobs create \
  --region=northamerica-northeast2 \
  --display-name=sanity-check-stage0 \
  --worker-pool-spec=machine-type=g2-standard-8,accelerator-type=NVIDIA_L4,accelerator-count=1,container-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest,local-package-path=vertex/package/sanity_check,python-module=sanity_check.cli,executor-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest,args="--checkpoint_gcs_uri=gs://liquid-llm-bucket-2/stage0/checkpoints/vertex_runs/20251009-022648/IMPORTANT/stage0_checkpoints_vertex_runs_20251009-022648_best.pt","--block_size=512","--device=cuda","--dtype=bfloat16","--throughput_tokens=32768","--batch_size=8"
```

## Exit Codes

* `0` – all required checks passed
* `2` – required checks failed (environment, checkpoint load, forward or backward tests)
* `1` – unexpected error (e.g., uncaught exception)

## Logging

The CLI prints newline-delimited JSON objects. Each event includes the package version, run identifier, device information, and metrics. The final line is a summary object with aggregated pass/fail counts followed by a human-readable PASS/FAIL line.
