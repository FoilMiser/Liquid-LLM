#!/usr/bin/env bash
set -euo pipefail
gsutil cp dist/liquid_llm_vertex_pkg_3-0.0.1-py3-none-any.whl gs://liquid-llm-bucket/vertex_pkg.whl
