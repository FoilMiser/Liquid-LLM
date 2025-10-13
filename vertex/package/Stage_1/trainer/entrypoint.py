"""Compatibility shim with deterministic dependency bootstrapping."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib import metadata
from typing import Iterable, Sequence

try:  # packaging is part of pip's vendored dependencies
    from packaging.requirements import Requirement
except Exception:  # pragma: no cover - fallback if packaging is missing
    Requirement = None  # type: ignore[assignment]


# Packages that must be present in the runtime before Stage-1 training can start.
# Torch is intentionally omitted from the install list because the Vertex base
# image already ships with the correct build; we only validate its version later
# when printing the dependency summary.
_REQUIRED_DEPENDENCIES: tuple[str, ...] = (
    "accelerate==1.10.1",
    "datasets==2.20.0",
    "einops>=0.8.0",
    "fsspec==2024.5.0",
    "gcsfs==2024.5.0",
    "google-cloud-secret-manager==2.24.0",
    "google-cloud-storage==2.18.2",
    "huggingface_hub>=0.23.2",
    "numpy>=1.26.4,<2.0",
    "pyarrow>=15.0.0,<16.0.0",
    "python-json-logger>=2.0.7",
    "safetensors>=0.6.2",
    "sentencepiece>=0.2.0",
    "tokenizers==0.22.1",
    "transformers==4.57.0",
    "tqdm>=4.66.4",
    "pyyaml>=6.0.2",
)

_SUMMARY_PACKAGES: tuple[str, ...] = (
    "torch",
    "transformers",
    "datasets",
    "pyarrow",
    "tokenizers",
    "numpy",
)

_BOOTSTRAPPED = False
_LOG_PREFIX = "[stage1-bootstrap]"


def _normalise_path_env() -> None:
    """Ensure ~/.local/bin is on PATH so pip console scripts work noiselessly."""

    local_bin = os.path.expanduser("~/.local/bin")
    path = os.environ.get("PATH", "")
    if local_bin not in path.split(":"):
        os.environ["PATH"] = f"{local_bin}:{path}" if path else local_bin


def _run(cmd: Sequence[str]) -> None:
    """Run a subprocess command with standardised pip environment settings."""

    env = os.environ.copy()
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("PIP_PROGRESS_BAR", "off")
    subprocess.run(cmd, check=True, env=env)


def _is_requirement_satisfied(spec: str) -> bool:
    if Requirement is None:  # packaging could only be missing in pathological cases
        return False

    requirement = Requirement(spec)
    try:
        resolved_version = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        return False

    # An empty specifier always means "any version"; otherwise confirm the
    # installed version lies within the expected range.
    return not requirement.specifier or resolved_version in requirement.specifier


def _gather_missing_requirements(specs: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for spec in specs:
        if not _is_requirement_satisfied(spec):
            missing.append(spec)
    return missing


def _print_dependency_summary() -> None:
    versions: list[str] = []
    for package in _SUMMARY_PACKAGES:
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "missing"
        versions.append(f"{package}={version}")
    print(f"{_LOG_PREFIX} resolved {'; '.join(versions)}")


def _bootstrap() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    _normalise_path_env()

    try:
        _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

        missing = _gather_missing_requirements(_REQUIRED_DEPENDENCIES)
        if missing:
            _run([sys.executable, "-m", "pip", "install", "--user", "--upgrade", *missing])

        _run([sys.executable, "-m", "pip", "check"])
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"{_LOG_PREFIX} pip failed with exit code {exc.returncode}: {exc}") from exc

    _print_dependency_summary()
    _BOOTSTRAPPED = True


_bootstrap()

# Import after bootstrapping so dependency resolution errors surface deterministically.
from Stage_1.vertex.entrypoint import *  # noqa: E402,F401,F403
