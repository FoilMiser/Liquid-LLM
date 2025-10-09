"""Vertex AI Stage-0 checkpoint sanity checking package."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("vertex-sanity-check")
except PackageNotFoundError:  # pragma: no cover
    from pathlib import Path

    _version_path = Path(__file__).with_name("VERSION")
    __version__ = _version_path.read_text().strip()

__all__ = ["__version__"]
