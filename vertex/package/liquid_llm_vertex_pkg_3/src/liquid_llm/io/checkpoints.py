import torch
from pathlib import Path
from .gcs import gcs_download_to, gcs_upload_from

def load_from_uri(model, optimizer, scheduler, uri: str):
    local = Path('/tmp/liquid_work/ckpt.pt')
    gcs_download_to(local, uri)
    return load_local(model, optimizer, scheduler, local)

def load_local(model, optimizer, scheduler, path: str | Path):
    sd = torch.load(path, map_location='cpu')
    # Flexible keys: either pure model sd or dict with fields
    if isinstance(sd, dict) and 'model' in sd:
        model.load_state_dict(sd['model'], strict=False)
        if optimizer is not None and 'optimizer' in sd:
            optimizer.load_state_dict(sd['optimizer'])
        if scheduler is not None and 'scheduler' in sd:
            scheduler.load_state_dict(sd['scheduler'])
        step = sd.get('step', 0)
    else:
        model.load_state_dict(sd, strict=False)
        step = 0
    return step

def save_local(state: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return str(path)

def save_and_maybe_upload(state: dict, local_path: str | Path, gcs_prefix: str | None, filename='ckpt.pt'):
    p = Path(local_path)
    p.mkdir(parents=True, exist_ok=True)
    out = p / filename
    save_local(state, out)
    if gcs_prefix:
        if not gcs_prefix.endswith('/'):
            gcs_prefix = gcs_prefix + '/'
        uri = gcs_prefix + filename
        gcs_upload_from(out, uri)
        return str(out), uri
    return str(out), None
