# liquid_llm_vertex_pkg_4

Vertex AI custom training package for Liquid‑LLM Stage 0.

## Knowledge distillation monitoring

This package adds step-level logging for knowledge distillation runs. When a
teacher model is configured, every training step emits a `[kd-step]` log with
the individual cross-entropy and distillation losses, KL divergences in both
directions, mean-squared error between teacher and student logits, and
statistical summaries of both logits tensors. Periodic training logs also echo
the latest teacher/student logits confirmation so that Vertex AI job logs make
it easy to monitor student-teacher divergence over time.

## Build & upload

```bash
python -m pip install build
python -m build
gsutil cp dist/liquid_llm_vertex_pkg_4-0.0.1-py3-none-any.whl gs://liquid-llm-bucket/vertex_pkg.whl
```

## Run (Vertex)

Set `python_module=trainer.entrypoint` and `package_uris=["gs://liquid-llm-bucket/vertex_pkg.whl"]`.

### Hugging Face token sources

The trainer now supports multiple token sources before falling back to
Secret Manager:

1. Pre-populated environment variables (`HUGGING_FACE_HUB_TOKEN` or `HF_TOKEN`).
2. `--hf_token_value="<token>"` passed on the command line or via YAML config.
3. `--hf_token_file=/path/to/token.txt` for a local file mounted into the job.
4. `--hf_token_gcs_uri=gs://bucket/path/to/token.txt` downloaded at runtime.
5. Secret Manager secrets specified by `--hf_secret_name` (and the default
   `hf_token` / `hf-token` fallbacks).

Explicit sources are a convenient workaround when the Vertex AI service
account lacks permission to access Secret Manager.
