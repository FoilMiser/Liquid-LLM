# Stage-1 Vertex AI Runbook

Recommended Vertex AI Custom Training configuration:

- **Worker pool spec**
  - Machine: `g2-standard-8`
  - Accelerator: `NVIDIA_L4` (count=1)
  - Container: `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest`
  - Local package path: `vertex/package/Stage_1`
  - Python module: `Stage_1.cli`

Paste the arguments from `arguments.txt` into the Vertex console Arguments box.

If you encounter OOMs or throughput is insufficient, upgrade to `a2-highgpu-1g` with an `NVIDIA_A100_40GB` accelerator using the same arguments.
