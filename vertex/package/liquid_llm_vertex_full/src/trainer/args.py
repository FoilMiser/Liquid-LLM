import argparse, yaml, os
from pathlib import Path

def get_parser():
    p = argparse.ArgumentParser(description="Liquid‑LLM Stage0 Vertex launcher")
    # Required-ish (as used in your run)
    p.add_argument('--resume_gcs_uri', type=str, default=None)
    p.add_argument('--block_size', type=int, required=True)
    p.add_argument('--teacher_name', type=str, required=True)
    p.add_argument('--dataset_name', type=str, required=True)
    p.add_argument('--dataset_config', type=str, required=True)
    p.add_argument('--hf_secret_name', type=str, default=None,
                   help='Secret Manager name containing the Hugging Face token')
    p.add_argument('--hf_token', type=str, default=None,
                   help='Direct Hugging Face token value (use Secret Manager when possible)')
    p.add_argument('--hf_token_file', type=str, default=None,
                   help='Local path to a file containing the Hugging Face token')
    p.add_argument('--hf_token_gcs_uri', type=str, default=None,
                   help='GCS URI to a file containing the Hugging Face token')

    # Outputs
    p.add_argument('--output_gcs_uri', type=str, default=None)
    p.add_argument('--local_workdir', type=str, default='/tmp/liquid_work')

    # Hyperparams / training knobs
    p.add_argument('--config', type=str, default=None, help='YAML file to merge as defaults')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--global_batch', type=int, default=None)
    p.add_argument('--micro_batch', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=None)
    p.add_argument('--betas', type=float, nargs=2, default=None)
    p.add_argument('--eps', type=float, default=None)
    p.add_argument('--warmup_steps', type=int, default=None)
    p.add_argument('--train_steps', type=int, default=None)
    p.add_argument('--eval_every', type=int, default=None)
    p.add_argument('--save_every', type=int, default=None)
    p.add_argument('--log_interval', type=int, default=None)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--bf16', action='store_true')

    # Model
    p.add_argument('--d_model', type=int, default=None)
    p.add_argument('--n_layers', type=int, default=None)
    p.add_argument('--n_heads', type=int, default=None)
    p.add_argument('--dropout', type=float, default=None)

    return p

def parse_args(argv=None):
    p = get_parser()
    args = p.parse_args(argv)

    # Merge config file if provided
    merged = {}
    if args.config:
        with open(args.config, 'r') as f:
            merged = yaml.safe_load(f) or {}

    # CLI overrides config
    def o(name, default=None):
        v = getattr(args, name, None)
        return default if v is None else v

    cfg = {
        'resume_gcs_uri': args.resume_gcs_uri,
        'block_size': args.block_size,
        'teacher_name': args.teacher_name,
        'dataset_name': args.dataset_name,
        'dataset_config': args.dataset_config,
        'output_gcs_uri': args.output_gcs_uri,
        'local_workdir': args.local_workdir,
        'hf_secret_name': o('hf_secret_name', merged.get('hf_secret_name')),
        'hf_token_value': o('hf_token', merged.get('hf_token_value')),
        'hf_token_file': o('hf_token_file', merged.get('hf_token_file')),
        'hf_token_gcs_uri': o('hf_token_gcs_uri', merged.get('hf_token_gcs_uri')),
        'seed': o('seed', merged.get('seed', 42)),
        'global_batch': o('global_batch', merged.get('global_batch', 64)),
        'micro_batch': o('micro_batch', merged.get('micro_batch', 8)),
        'lr': o('lr', merged.get('lr', 3e-4)),
        'weight_decay': o('weight_decay', merged.get('weight_decay', 0.1)),
        'betas': tuple(o('betas', merged.get('betas', [0.9, 0.95]))),
        'eps': o('eps', merged.get('eps', 1e-8)),
        'warmup_steps': o('warmup_steps', merged.get('warmup_steps', 2000)),
        'train_steps': o('train_steps', merged.get('train_steps', 10000)),
        'eval_every': o('eval_every', merged.get('eval_every', 500)),
        'save_every': o('save_every', merged.get('save_every', 1000)),
        'log_interval': o('log_interval', merged.get('log_interval', 50)),
        'precision': 'bf16' if args.bf16 else ('fp16' if args.fp16 else 'no'),
        'model': {
            'd_model': o('d_model', merged.get('model', {}).get('d_model', 768)),
            'n_layers': o('n_layers', merged.get('model', {}).get('n_layers', 10)),
            'n_heads': o('n_heads', merged.get('model', {}).get('n_heads', 12)),
            'dropout': o('dropout', merged.get('model', {}).get('dropout', 0.0)),
        }
    }
    return cfg
