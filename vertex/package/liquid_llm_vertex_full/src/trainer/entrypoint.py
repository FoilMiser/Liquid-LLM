import os, sys
from pathlib import Path

from .args import parse_args
from .secrets import ensure_hf_token
from liquid_llm.utils.logging import init_logger, get_logger
from liquid_llm.utils.seed import set_all_seeds
from liquid_llm.training.stage0 import run_training

def main(argv=None):
    cfg = parse_args(argv)
    Path(cfg['local_workdir']).mkdir(parents=True, exist_ok=True)
    init_logger()
    log = get_logger('entrypoint')

    sensitive_keys = {"hf_token_value", "hf_token_file", "hf_token_gcs_uri"}
    redacted_cfg = {
        key: ("<redacted>" if key in sensitive_keys and cfg.get(key) else value)
        for key, value in cfg.items()
    }
    log.info("Parsed config: %s", redacted_cfg)
    set_all_seeds(cfg['seed'])

    secret_names = None
    if cfg.get('hf_secret_name'):
        secret_names = [cfg['hf_secret_name']]
    hf_token = ensure_hf_token(
        secret_names=secret_names,
        explicit_token=cfg.pop('hf_token_value', None),
        token_file=cfg.pop('hf_token_file', None),
        token_gcs_uri=cfg.pop('hf_token_gcs_uri', None),
        log=log,
    )
    cfg['hf_token'] = hf_token

    run_training(**cfg)

if __name__ == '__main__':
    main()
