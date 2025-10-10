"""Secret manager helpers."""

from __future__ import annotations

import json
from typing import Optional

try:
    from google.cloud import secretmanager  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    secretmanager = None


def get_secret(secret_name: str, project_id: Optional[str] = None) -> Optional[str]:
    """Fetch a secret from Google Secret Manager if available."""

    if not secret_name:
        return None
    if secretmanager is None:
        return None
    client = secretmanager.SecretManagerServiceClient()
    if "/" in secret_name:
        name = secret_name
    else:
        if project_id is None:
            raise ValueError("project_id required when secret path not fully qualified")
        name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(name=name)
    payload = response.payload.data.decode("UTF-8")
    try:
        data = json.loads(payload)
        token = data.get("token") or data.get("value")
        if token:
            return token
    except json.JSONDecodeError:
        pass
    return payload


__all__ = ["get_secret"]
