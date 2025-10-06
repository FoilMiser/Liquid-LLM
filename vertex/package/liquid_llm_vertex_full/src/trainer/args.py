import argparse
import yaml

def get_parser():
    p = argparse.ArgumentParser(description="Liquid‑LLM Stage0 Vertex launcher")
    # Required-ish (as used in your run)
    p.add_argument('--resume_gcs_uri', type=str, default=None)
    p.add_argument('--block_size', type=int, required=True)
    p.add_argument('--teacher_name', type=str, required=True)
    p.add_argument('--dataset_name', type=str, required=True)
    p.add_argument('--dataset_config', type=str, required=True)

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
    p.add_argument('--grad_clip', type=float, default=None)
    p.add_argument('--kd_alpha', type=float, default=None)
    p.add_argument('--kd_temperature', type=float, default=None)
    p.add_argument('--teacher_eval_every', type=int, default=None)

    # Model
    p.add_argument('--d_model', type=int, default=None)
    p.add_argument('--n_layers', type=int, default=None)
    p.add_argument('--n_heads', type=int, default=None)
    p.add_argument('--dropout', type=float, default=None)

    # Checkpoint retention
    p.add_argument('--time_ckpt_secs', type=int, default=None)
    p.add_argument('--time_ckpt_retention_secs', type=int, default=None)
    p.add_argument('--time_ckpt_keep_k', type=int, default=None)
    p.add_argument('--best_ckpt_keep_k', type=int, default=None)
    p.add_argument('--best_ckpt_retention_secs', type=int, default=None)
    p.add_argument('--step_ckpt_keep_k', type=int, default=None)
    p.add_argument('--step_ckpt_retention_secs', type=int, default=None)

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
        'grad_clip': o('grad_clip', merged.get('grad_clip', 1.0)),
        'kd_alpha': o('kd_alpha', merged.get('kd_alpha', 0.5)),
        'kd_temperature': o('kd_temperature', merged.get('kd_temperature', 1.0)),
        'teacher_eval_every': o('teacher_eval_every', merged.get('teacher_eval_every', 0)),
        'time_ckpt_secs': o('time_ckpt_secs', merged.get('time_ckpt_secs', 1800)),
        'time_ckpt_retention_secs': o('time_ckpt_retention_secs', merged.get('time_ckpt_retention_secs', 14400)),
        'time_ckpt_keep_k': o('time_ckpt_keep_k', merged.get('time_ckpt_keep_k', None)),
        'best_ckpt_keep_k': o('best_ckpt_keep_k', merged.get('best_ckpt_keep_k', 3)),
        'best_ckpt_retention_secs': o('best_ckpt_retention_secs', merged.get('best_ckpt_retention_secs', None)),
        'step_ckpt_keep_k': o('step_ckpt_keep_k', merged.get('step_ckpt_keep_k', 5)),
        'step_ckpt_retention_secs': o('step_ckpt_retention_secs', merged.get('step_ckpt_retention_secs', None)),
        'model': {
            'd_model': o('d_model', merged.get('model', {}).get('d_model', 768)),
            'n_layers': o('n_layers', merged.get('model', {}).get('n_layers', 10)),
            'n_heads': o('n_heads', merged.get('model', {}).get('n_heads', 12)),
            'dropout': o('dropout', merged.get('model', {}).get('dropout', 0.0)),
        }
    }
    return cfg
