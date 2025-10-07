from google.cloud import storage
from pathlib import Path
import os

def parse_gcs_uri(uri: str):
    assert uri.startswith('gs://'), f'Not a GCS URI: {uri}'
    no_scheme = uri[5:]
    bucket, *rest = no_scheme.split('/', 1)
    prefix = rest[0] if rest else ''
    return bucket, prefix

def gcs_download_to(local_path: str | Path, gcs_uri: str):
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    return str(local_path)

def gcs_upload_from(local_path: str | Path, gcs_uri: str, overwrite=True):
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if (not overwrite) and blob.exists(client):
        raise RuntimeError(f'Blob already exists: {gcs_uri}')
    blob.upload_from_filename(str(local_path))
    return gcs_uri
