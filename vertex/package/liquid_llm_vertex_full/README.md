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

## Troubleshooting

### Hugging Face token failures

The training entrypoint resolves the Hugging Face token in the following order:

1. `HUGGING_FACE_HUB_TOKEN` / `HF_TOKEN` environment variables already present on the worker.
2. A token supplied explicitly via `--hf_token`, `--hf_token_file`, or
   `--hf_token_gcs_uri`.
3. Secrets fetched from Secret Manager using the names supplied via
   `--hf_secret_name` (defaulting to `hf-token`) and the built-in fallbacks
   `hf_token` and `hf-token`.

If none of those lookups succeed the job will exit with an error similar to:

```
RuntimeError: Failed to access any configured Hugging Face token secret. Tried: hf_token, hf-token (details: ... ACCESS_TOKEN_SCOPE_INSUFFICIENT ...)
```

This indicates the Vertex AI service account or VM scopes cannot access the
Secret Manager secret. To resolve the issue:

* Grant the Vertex AI custom training service account the **Secret Manager
  Secret Accessor** role on the relevant secret (or the project containing it).
* Ensure the worker pool has the `cloud-platform` OAuth scope. When launching
  jobs via the API, include
 `worker_pool_specs[*].machine_spec.service_account` (if using a custom service
  account).
* If the service account cannot read from Secret Manager, provide the token via
  `worker_pool_specs[*].container_spec.env` or by referencing a file using
  `--hf_token_file` / `--hf_token_gcs_uri`.

When using the CLI flags directly, prefer `--hf_token_file` or
`--hf_token_gcs_uri` so that the token can be managed outside of command-line
arguments. The direct `--hf_token` flag is available for completeness but should
only be used for short-lived experiments.

After permissions are in place, retry the job or provide the token via
environment variables.
