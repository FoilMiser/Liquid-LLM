# liquid_llm_vertex

Vertex AI custom training package for Liquid‑LLM Stage 0.

## Build & upload

```bash
python -m pip install build
python -m build
gsutil cp dist/liquid_llm_vertex-0.0.1-py3-none-any.whl gs://liquid-llm-bucket/vertex_pkg.whl
```

## Run (Vertex)

Set `python_module=trainer.entrypoint` and `package_uris=["gs://liquid-llm-bucket/vertex_pkg.whl"]`.
