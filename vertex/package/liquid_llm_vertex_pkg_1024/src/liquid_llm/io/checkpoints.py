import json
import torch
from pathlib import Path
from typing import Dict, Any, Tuple
from .gcs import gcs_download_to, gcs_upload_from


def _ensure_tmp_dir() -> Path:
    path = Path('/tmp/liquid_work')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _meta_path_for(local_path: Path) -> Path:
    return local_path.parent / f"{local_path.name}.meta.json"


def load_from_uri(model, optimizer, scheduler, uri: str) -> Tuple[int, Dict[str, Any]]:
    tmp_dir = _ensure_tmp_dir()
    filename = Path(uri).name or 'ckpt.pt'
    local = tmp_dir / filename
    gcs_download_to(local, uri)
    meta_uri = uri + '.meta.json'
    meta_local = _meta_path_for(local)
    try:
        gcs_download_to(meta_local, meta_uri)
    except Exception:
        meta_local = None
    return load_local(model, optimizer, scheduler, local, meta_local)


def load_local(model, optimizer, scheduler, path: str | Path, meta_path: str | Path | None = None) -> Tuple[int, Dict[str, Any]]:
    path = Path(path)
    sd = torch.load(path, map_location='cpu')
    trainer_state = {}
    meta = None
    # Flexible keys: either pure model sd or dict with fields
    if isinstance(sd, dict) and 'model' in sd:
        model.load_state_dict(sd['model'], strict=False)
        if optimizer is not None and 'optimizer' in sd:
            optimizer.load_state_dict(sd['optimizer'])
        if scheduler is not None and 'scheduler' in sd:
            scheduler.load_state_dict(sd['scheduler'])
        step = sd.get('step', 0)
        trainer_state = sd.get('trainer_state', {}) or {}
        meta = sd.get('meta')
    else:
        model.load_state_dict(sd, strict=False)
        step = 0
    if meta is None:
        if meta_path is None:
            meta_path = _meta_path_for(path)
        meta_path = Path(meta_path)
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = None
    return step, {'trainer_state': trainer_state, 'meta': meta}


def save_local(state: dict, path: str | Path, meta: Dict[str, Any] | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    meta_path = None
    if meta is not None:
        meta_path = _meta_path_for(path)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, sort_keys=True)
    return str(path), (str(meta_path) if meta_path else None)


def save_and_maybe_upload(state: dict, local_path: str | Path, gcs_prefix: str | None, filename='ckpt.pt', meta: Dict[str, Any] | None = None):
    p = Path(local_path)
    p.mkdir(parents=True, exist_ok=True)
    out = p / filename
    saved_path, meta_path = save_local(state, out, meta=meta)
    remote_uri = None
    if gcs_prefix:
        if not gcs_prefix.endswith('/'):
            gcs_prefix = gcs_prefix + '/'
        uri = gcs_prefix + filename
        gcs_upload_from(out, uri)
        remote_uri = uri
        if meta_path:
            gcs_upload_from(meta_path, uri + '.meta.json')
    return saved_path, remote_uri
