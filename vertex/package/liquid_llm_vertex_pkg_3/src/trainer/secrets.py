"""Utilities for fetching secrets needed by the Vertex trainer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from google.cloud import secretmanager, storage


_PROJECT_ENV_VARS: Iterable[str] = (
    "AIP_PROJECT_NUMBER",
    "AIP_PROJECT_ID",
    "GOOGLE_CLOUD_PROJECT",
    "CLOUD_ML_PROJECT_ID",
    "GCP_PROJECT",
    "PROJECT_ID",
)


def _detect_project_id() -> Optional[str]:
    """Best effort detection of the active Google Cloud project ID/number."""

    for key in _PROJECT_ENV_VARS:
        value = os.environ.get(key)
        if value:
            return value
    return None


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Unsupported URI scheme in '{uri}'. Expected gs://")

    remainder = uri[5:]
    if not remainder:
        raise ValueError(f"Missing bucket/object in '{uri}'")

    parts = remainder.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid GCS URI '{uri}'. Expected gs://bucket/object")

    return parts[0], parts[1]


def ensure_hf_token(
    *,
    secret_name: str = "hf-token",
    secret_names: Optional[Iterable[str]] = None,
    env_var: str = "HUGGING_FACE_HUB_TOKEN",
    token_value: Optional[str] = None,
    token_file: Optional[str] = None,
    token_gcs_uri: Optional[str] = None,
    storage_client: Optional[storage.Client] = None,
    log=None,
) -> str:
    """Ensure a Hugging Face token is available via env vars.

    The precedence order is:

    1. Existing environment variables (``HUGGING_FACE_HUB_TOKEN``/``HF_TOKEN``)
    2. Explicit token values via ``--hf_token_value``
    3. Local files referenced by ``--hf_token_file``
    4. GCS files referenced by ``--hf_token_gcs_uri``
    5. Google Secret Manager secrets in ``secret_names``/``secret_name`` order

    Args:
        secret_name: Default Secret Manager identifier containing the HF access token.
        secret_names: Optional iterable of secret names to try (in order).
        env_var: Environment variable name to populate with the token.
        log: Optional logger with an ``info``/``warning`` method.

    Returns:
        The Hugging Face token string.

    Raises:
        RuntimeError: If the token cannot be resolved.
    """

    errors: list[tuple[str, Exception]] = []

    existing = os.environ.get(env_var) or os.environ.get("HF_TOKEN")
    if existing:
        token = existing.strip()
        os.environ.setdefault(env_var, token)
        os.environ.setdefault("HF_TOKEN", token)
        if log:
            log.info("Using Hugging Face token from environment variable.")
        return token

    def _record_failure(source: str, exc: Exception) -> None:
        errors.append((source, exc))

    def _use_token(raw: str | None, source: str) -> Optional[str]:
        if raw is None:
            return None
        token = raw.strip()
        if not token:
            _record_failure(source, ValueError("token payload empty"))
            return None
        os.environ[env_var] = token
        os.environ.setdefault("HF_TOKEN", token)
        if log:
            log.info("Loaded Hugging Face token from %s.", source)
        return token

    token = _use_token(token_value, "explicit --hf_token_value")
    if token:
        return token

    if token_file:
        path = Path(os.path.expandvars(os.path.expanduser(token_file)))
        try:
            contents = path.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - filesystem dependent
            _record_failure(f"file:{token_file}", exc)
        else:
            token = _use_token(contents, f"local file '{token_file}'")
            if token:
                return token

    if token_gcs_uri:
        try:
            bucket_name, object_name = _parse_gcs_uri(token_gcs_uri)
            client = storage_client or storage.Client()
            blob = client.bucket(bucket_name).blob(object_name)
            contents = blob.download_as_text().strip()
            token = _use_token(contents, f"GCS URI '{token_gcs_uri}'")
            if token:
                return token
        except Exception as exc:  # pragma: no cover - network/cloud dependent
            _record_failure(f"gcs:{token_gcs_uri}", exc)

    project_id = _detect_project_id()
    if not project_id:
        details = ", ".join(f"{name}: {err}" for name, err in errors) or "no explicit sources succeeded"
        raise RuntimeError(
            "Unable to determine Google Cloud project for Secret Manager. "
            "Set the HUGGING_FACE_HUB_TOKEN environment variable or configure project metadata. "
            f"Previous attempts: {details}"
        )

    candidates = list(secret_names or ())
    if not candidates:
        candidates.append(secret_name)
    else:
        # maintain backwards compatibility with the legacy default name if a
        # single override was supplied.
        if len(candidates) == 1 and candidates[0] in {"hf-token", "hf_token"}:
            alt = "hf-token" if candidates[0] == "hf_token" else "hf_token"
            candidates.append(alt)

    # Ensure the default names are always attempted if not explicitly provided.
    for fallback in ("hf_token", "hf-token"):
        if fallback not in candidates:
            candidates.append(fallback)

    client = secretmanager.SecretManagerServiceClient()
    for candidate in candidates:
        if not candidate:
            continue
        name = f"projects/{project_id}/secrets/{candidate}/versions/latest"
        try:
            response = client.access_secret_version(name=name)
        except Exception as exc:  # pragma: no cover - bubble up with context
            _record_failure(candidate, exc)
            continue

        secret_source = (
            f"Secret Manager secret '{candidate}' (project '{project_id}')"
        )
        token = _use_token(response.payload.data.decode("utf-8"), secret_source)
        if token:
            return token

    details = ", ".join(f"{name}: {err}" for name, err in errors) or "no candidates"
    raise RuntimeError(
        "Failed to resolve a Hugging Face token from any configured source. "
        f"Attempts: {details}"
    )

