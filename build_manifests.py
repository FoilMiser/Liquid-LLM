"""Utility for building Stage-1 dataset manifest files for Vertex AI training/precompute."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import yaml


DEFAULT_DATASETS_YAML = "preprocess_toolkit/config/datasets.yaml"
DEFAULT_OUT_DIR = "vertex/package/sandbox/manifest"
DEFAULT_BUCKET_PREFIX = "gs://liquid-llm-bucket-2/datasets/stage1/manifests"


class ManifestError(RuntimeError):
    """Raised when there is an issue constructing a manifest entry."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-yaml",
        default=DEFAULT_DATASETS_YAML,
        help="Path to the preprocess toolkit datasets YAML configuration.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Directory where manifest files will be written.",
    )
    parser.add_argument(
        "--bucket-prefix",
        default=DEFAULT_BUCKET_PREFIX,
        help="GCS prefix for manifest uploads.",
    )
    return parser.parse_args(argv)


def resolve_datasets_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    if not path.is_absolute():
        alternative = Path("vertex/package/sandbox/preprocess-toolkit") / path
        if alternative.exists():
            return alternative
    raise ManifestError(
        f"Datasets configuration file not found at '{path_str}'. "
        "Use --datasets-yaml to provide the correct path."
    )


ManifestEntry = Dict[str, object]


def parse_manifest_payload(raw: object) -> MutableMapping[str, object]:
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"Failed to parse manifest JSON: {exc}") from exc
        if not isinstance(parsed, MutableMapping):
            raise ManifestError("Parsed manifest JSON must be an object/dict.")
        return parsed
    if isinstance(raw, MutableMapping):
        return raw
    raise ManifestError("Manifest payload must be a JSON string or mapping.")


def normalize_weight(value: object | None) -> float:
    weight = 1.0 if value is None else value
    try:
        return float(weight)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"Manifest weight must be numeric, got: {value!r}") from exc


def build_entry_from_yaml(job: str, payload: Mapping[str, object]) -> ManifestEntry:
    if not isinstance(payload, Mapping):
        raise ManifestError(f"Dataset '{job}' payload must be a mapping.")

    manifest_field = payload.get("manifest")
    if manifest_field is not None:
        manifest_payload = parse_manifest_payload(manifest_field)
        entry: ManifestEntry = {}
        path = manifest_payload.get("path")
        if not isinstance(path, str) or not path:
            raise ManifestError(f"Dataset '{job}' manifest must include a non-empty 'path'.")
        entry["path"] = path

        type_value = manifest_payload.get("type", payload.get("type", "lm"))
        if not isinstance(type_value, str) or not type_value.strip():
            raise ManifestError(f"Dataset '{job}' manifest must include a string 'type'.")
        entry["type"] = type_value.strip().lower()

        entry["weight"] = normalize_weight(manifest_payload.get("weight"))
        return entry

    out_value = payload.get("out")
    if not isinstance(out_value, str) or not out_value.strip():
        raise ManifestError(
            f"Dataset '{job}' is missing required 'out' field needed to build manifest entry."
        )
    out_value = out_value.strip()
    if not out_value.startswith("gs://"):
        raise ManifestError(
            f"Dataset '{job}' has out='{out_value}', expected it to start with 'gs://'."
        )

    entry = {
        "path": f"{out_value.rstrip('/')}/*.jsonl",
        "type": str(payload.get("type", "lm")).strip().lower() or "lm",
        "weight": 1.0,
    }
    return entry


def write_jsonl(path: Path, entries: Iterable[ManifestEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, separators=(",", ":")))
            fh.write("\n")


def filter_by_type(items: Sequence[tuple[str, ManifestEntry]], allowed_types: set[str]) -> List[ManifestEntry]:
    return [entry for _, entry in items if entry["type"] in allowed_types]


def to_windows_path(path: Path) -> str:
    rel_path = path.as_posix()
    return ".\\" + rel_path.replace("/", "\\")


def print_upload_command(local_path: Path, gcs_uri: str) -> None:
    print(
        "gcloud storage cp --no-clobber "
        f'"{to_windows_path(local_path)}" "{gcs_uri}"'
    )


def preview_entries(name: str, entries: Sequence[ManifestEntry]) -> None:
    print(f"Preview {name}:")
    if not entries:
        print("(empty)")
        return
    for entry in entries[:2]:
        print(json.dumps(entry, separators=(",", ":")))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        datasets_path = resolve_datasets_path(args.datasets_yaml)
    except ManifestError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    with datasets_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if not isinstance(data, Mapping):
        print(
            "Datasets YAML must define a mapping from job names to dataset configs.",
            file=sys.stderr,
        )
        return 1

    entries_by_job: Dict[str, ManifestEntry] = {}
    for job, payload in data.items():
        if not isinstance(job, str):
            print(f"Skipping non-string job key: {job!r}", file=sys.stderr)
            continue
        try:
            entry = build_entry_from_yaml(job, payload)
        except ManifestError as exc:
            print(f"Error processing dataset '{job}': {exc}", file=sys.stderr)
            return 1
        entries_by_job[job] = entry

    sorted_items = sorted(entries_by_job.items())
    stage1_entries = [entry for _, entry in sorted_items]
    stage1_lm_entries = filter_by_type(sorted_items, {"lm"})
    stage1_math_code_entries = filter_by_type(sorted_items, {"math_tool", "code"})

    out_dir = Path(args.out_dir)
    stage1_path = out_dir / "stage1.jsonl"
    stage1_lm_path = out_dir / "stage1_lm.jsonl"
    stage1_math_code_path = out_dir / "stage1_math_code.jsonl"

    write_jsonl(stage1_path, stage1_entries)
    write_jsonl(stage1_lm_path, stage1_lm_entries)
    write_jsonl(stage1_math_code_path, stage1_math_code_entries)

    by_dataset_dir = out_dir / "by_dataset"
    for job, entry in sorted_items:
        write_jsonl(by_dataset_dir / f"{job}.jsonl", [entry])

    counts = Counter(entry["type"] for entry in stage1_entries)
    print(
        "Counts: total={total}, lm={lm}, math_tool={math_tool}, code={code}".format(
            total=len(stage1_entries),
            lm=counts.get("lm", 0),
            math_tool=counts.get("math_tool", 0),
            code=counts.get("code", 0),
        )
    )

    preview_entries("stage1.jsonl", stage1_entries)
    preview_entries("stage1_lm.jsonl", stage1_lm_entries)
    preview_entries("stage1_math_code.jsonl", stage1_math_code_entries)

    bucket_prefix = args.bucket_prefix.rstrip("/")
    print_upload_command(stage1_path, f"{bucket_prefix}/stage1.jsonl")
    print_upload_command(stage1_lm_path, f"{bucket_prefix}/stage1_lm.jsonl")
    print_upload_command(stage1_math_code_path, f"{bucket_prefix}/stage1_math_code.jsonl")

    for job in sorted(entries_by_job):
        local_path = by_dataset_dir / f"{job}.jsonl"
        gcs_uri = f"{bucket_prefix}/by_dataset/{job}.jsonl"
        print_upload_command(local_path, gcs_uri)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
