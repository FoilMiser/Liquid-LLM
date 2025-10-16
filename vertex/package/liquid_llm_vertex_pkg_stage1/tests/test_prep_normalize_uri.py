"""Tests for preprocess bootstrap URI helpers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1 import prep


def test_normalize_uri_adds_prefix():
    uri = "liquid-llm-bucket-2/path/to/toolkit.zip"
    assert prep.normalize_gcs_uri(uri) == f"gs://{uri}"


def test_normalize_uri_preserves_prefix():
    uri = "gs://bucket/path/file.zip"
    assert prep.normalize_gcs_uri(uri) == uri


def test_normalize_uri_default():
    result = prep.normalize_gcs_uri(None)
    assert result.startswith("gs://")
