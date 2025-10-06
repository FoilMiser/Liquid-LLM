"""Utilities for fetching secrets needed by the Vertex trainer."""

from __future__ import annotations

import os
from typing import Iterable, Optional

import google.auth
from google.api_core import exceptions as gcloud_exceptions
from google.auth import exceptions as auth_exceptions
from google.cloud import secretmanager


_PROJECT_ENV_VARS: Iterable[str] = (
    "AIP_PROJECT_NUMBER",
    "AIP_PROJECT_ID",
    "GOOGLE_CLOUD_PROJECT",
    "CLOUD_ML_PROJECT_ID",
    "GCP_PROJECT",
    "PROJECT_ID",
)


_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def _detect_project_id() -> Optional[str]:
    """Best effort detection of the active Google Cloud project ID/number."""

    for key in _PROJECT_ENV_VARS:
        value = os.environ.get(key)
        if value:
            return value
    return None


def _build_secret_client(log=None):
    """Construct a Secret Manager client with cloud-platform scope if possible."""

    credentials = None
    project_from_creds: Optional[str] = None
    try:
        credentials, project_from_creds = google.auth.default(scopes=(_CLOUD_PLATFORM_SCOPE,))
    except auth_exceptions.DefaultCredentialsError as exc:
        if log:
            log.warning("Unable to load default Google credentials: %s", exc)

    client = (
        secretmanager.SecretManagerServiceClient(credentials=credentials)
        if credentials is not None
        else secretmanager.SecretManagerServiceClient()
    )

    return client, project_from_creds


def ensure_hf_token(
    *,
    secret_name: str = "hf-token",
    secret_names: Optional[Iterable[str]] = None,
    env_var: str = "HUGGING_FACE_HUB_TOKEN",
    log=None,
) -> str:
    """Ensure a Hugging Face token is available via env vars.

    The precedence order is:

    1. Existing environment variables (``HUGGING_FACE_HUB_TOKEN``/``HF_TOKEN``)
    2. Google Secret Manager secrets in ``secret_names``/``secret_name`` order

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

    existing = os.environ.get(env_var) or os.environ.get("HF_TOKEN")
    if existing:
        token = existing.strip()
        if log:
            log.info("Using Hugging Face token from environment variable.")
        return token

    client, project_from_creds = _build_secret_client(log=log)

    project_id = _detect_project_id() or project_from_creds
    if log and project_from_creds and project_id == project_from_creds:
        log.info("Resolved Google Cloud project from application default credentials: %s", project_id)
    if not project_id:
        raise RuntimeError(
            "Unable to determine Google Cloud project for Secret Manager. "
            "Set the HUGGING_FACE_HUB_TOKEN environment variable or configure project metadata."
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

    # Remove duplicates while preserving order so we don't spam Secret Manager or logs.
    deduped_candidates = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        deduped_candidates.append(candidate)
        seen.add(candidate)
    candidates = deduped_candidates

    errors = []
    for candidate in candidates:
        if not candidate:
            continue
        if log:
            log.info(
                "Attempting to access Hugging Face token from Secret Manager secret '%s'.",
                candidate,
            )
        name = f"projects/{project_id}/secrets/{candidate}/versions/latest"
        try:
            response = client.access_secret_version(name=name)
        except Exception as exc:  # pragma: no cover - bubble up with context
            if log and isinstance(exc, gcloud_exceptions.PermissionDenied):
                log.error(
                    "Permission denied when accessing secret '%s'. Ensure the Vertex AI service "
                    "account has Secret Manager Secret Accessor roles and the VM has the "
                    "cloud-platform OAuth scope.",
                    candidate,
                )
            errors.append((candidate, exc))
            continue

        token = response.payload.data.decode("utf-8").strip()
        if not token:
            errors.append((candidate, ValueError("secret payload empty")))
            continue

        os.environ[env_var] = token
        os.environ.setdefault("HF_TOKEN", token)

        if log:
            log.info(
                "Loaded Hugging Face token from Secret Manager secret '%s' (project '%s').",
                candidate,
                project_id,
            )

        return token

    details = ", ".join(f"{name}: {err}" for name, err in errors) or "no candidates"
    raise RuntimeError(
        "Failed to access any configured Hugging Face token secret. "
        f"Tried: {', '.join(candidates)} (details: {details})"
    )

