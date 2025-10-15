# Preprocess Toolkit

This toolkit provides dataset preprocessing pipelines that normalize diverse input
formats into standardized JSONL shards ready for model training. Pipelines run in
Vertex containers or on local Linux machines and stream data to minimize memory
usage.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run an individual dataset by supplying the dataset name, input URI, and output
shard directory. For example, to process WikiText-103:

```bash
python -m preprocess_toolkit.driver \
  --job wikitext \
  --in gs://liquid-llm-bucket-2/datasets/stage1/core/wikitext/wikitext-103-v1.zip \
  --out gs://liquid-llm-bucket-2/datasets/stage1/core/wikitext/shards/ \
  --max-records 20000 --manifest
```

Run all configured datasets from `config/datasets.yaml`:

```bash
python -m preprocess_toolkit.driver --job all --manifest
```

## Notes

* All pipelines stream their inputs and write incremental shards to avoid high
  memory usage. Large files are downloaded to the temporary working directory
  and processed chunk-by-chunk.
* GCS operations use `gcloud storage` with retries. If unavailable, the toolkit
  falls back to `gcsfs` for object access.
* To adjust dataset sampling weights or shard destinations, edit
  `config/datasets.yaml` and rerun the manifest builder or pipeline driver.
* The manifest builder prints manifest lines exactly as expected by downstream
  consumers, making it easy to regenerate manifests after updates.
