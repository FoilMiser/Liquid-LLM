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

    log.info(f"Parsed config: {cfg}")
    set_all_seeds(cfg['seed'])

    hf_token = ensure_hf_token(log=log)
    cfg['hf_token'] = hf_token

    run_training(**cfg)

if __name__ == '__main__':
    main()
