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
    p.add_argument('--hf_token_value', type=str, default=None,
                   help='Explicit Hugging Face token value (takes precedence over secrets)')
    p.add_argument('--hf_token_file', type=str, default=None,
                   help='Path to a file containing the Hugging Face token')
    p.add_argument('--hf_token_gcs_uri', type=str, default=None,
                   help='GCS URI to a file containing the Hugging Face token')
    p.add_argument('--require_hf_token', action='store_true',
                   help='Fail if a Hugging Face token cannot be resolved')

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
    p.add_argument('--lr_peak', type=float, default=None,
                   help='Optional peak learning rate to reach at the end of warmup')
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--teacher_precision', type=str, default=None,
                   help='Precision hint for the teacher model (e.g. fp16, bf16, fp32)')

    # Model
    p.add_argument('--d_model', type=int, default=None)
    p.add_argument('--n_layers', type=int, default=None)
    p.add_argument('--n_heads', type=int, default=None)
    p.add_argument('--dropout', type=float, default=None)

    # KD annealing / context management
    p.add_argument('--alpha', type=float, default=None,
                   help='Override initial KD alpha (defaults to checkpoint or 0.5)')
    p.add_argument('--T', type=float, default=None,
                   help='Override initial KD temperature (defaults to checkpoint or 1.0)')
    p.add_argument('--alpha_schedule', type=str, default=None,
                   help='Scalar schedule for KD alpha (e.g. cosine:hold_steps=2000,start=0.2,end=0.05,total_steps=25000)')
    p.add_argument('--temp_schedule', type=str, default=None,
                   help='Scalar schedule for KD temperature (e.g. linear:start=1.5,end=1.0,total_steps=10000)')
    p.add_argument('--kd_scheme', type=str, default=None,
                   help='Knowledge distillation scheme identifier (e.g. forward_kl)')
    p.add_argument('--kl_scale', type=str, default=None,
                   help='Scaling to apply to the KL loss (e.g. T^2, T, 1, or numeric)')
    p.add_argument('--save_best_on', type=str, default='val_loss_student',
                   help='Validation metric key used for best checkpointing (e.g. val_loss_student@1024)')
    p.add_argument('--save_every_steps', type=int, default=5000,
                   help='Save a step checkpoint every N global steps (0 disables)')
    p.add_argument('--schedule_from_zero', action='store_true',
                   help='Restart alpha/T/LR schedules from step zero when resuming')
    p.add_argument('--reset_lrsched_on_resume', action='store_true',
                   help='Reinitialize LR scheduler on resume to respect warmup from zero')
    p.add_argument('--eval_ctx_lens', type=str, default=None,
                   help='Comma separated context lengths to evaluate (e.g. 512,1024)')
    p.add_argument('--reset_optim_on_ctx_change', action='store_true',
                   help='Reinitialize optimizer/scheduler if resumed block size differs from checkpoint')
    p.add_argument('--grad_accum', type=int, default=None,
                   help='Number of micro batches to accumulate before each optimizer step')
    p.add_argument('--rope_scale', type=float, default=None,
                   help='Optional RoPE scaling factor for models that support rotary embeddings')

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

    require_hf_token = bool(merged.get('require_hf_token', False))
    if args.require_hf_token:
        require_hf_token = True

    def parse_eval_ctx_lens(value):
        if not value:
            return []
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        out = []
        for item in value.split(','):
            item = item.strip()
            if not item:
                continue
            out.append(int(item))
        return out

    def schedule_spec(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return ','.join(f"{int(step)}:{float(val)}" for step, val in value)
        if isinstance(value, dict):
            entries = sorted(value.items(), key=lambda x: int(x[0]))
            return ','.join(f"{int(step)}:{float(val)}" for step, val in entries)
        return str(value)

    cfg = {
        'resume_gcs_uri': args.resume_gcs_uri,
        'block_size': args.block_size,
        'teacher_name': args.teacher_name,
        'dataset_name': args.dataset_name,
        'dataset_config': args.dataset_config,
        'output_gcs_uri': args.output_gcs_uri,
        'local_workdir': args.local_workdir,
        'hf_secret_name': o('hf_secret_name', merged.get('hf_secret_name')),
        'hf_token_value': o('hf_token_value', merged.get('hf_token_value')),
        'hf_token_file': o('hf_token_file', merged.get('hf_token_file')),
        'hf_token_gcs_uri': o('hf_token_gcs_uri', merged.get('hf_token_gcs_uri')),
        'require_hf_token': require_hf_token,
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
        'lr_peak': o('lr_peak', merged.get('lr_peak')),
        'precision': 'bf16' if args.bf16 else ('fp16' if args.fp16 else 'no'),
        'teacher_precision': o('teacher_precision', merged.get('teacher_precision')),
        'model': {
            'd_model': o('d_model', merged.get('model', {}).get('d_model', 768)),
            'n_layers': o('n_layers', merged.get('model', {}).get('n_layers', 10)),
            'n_heads': o('n_heads', merged.get('model', {}).get('n_heads', 12)),
            'dropout': o('dropout', merged.get('model', {}).get('dropout', 0.0)),
        },
        'alpha': o('alpha', merged.get('alpha')),
        'T': o('T', merged.get('T')),
        'alpha_schedule': schedule_spec(o('alpha_schedule', merged.get('alpha_schedule'))),
        'temp_schedule': schedule_spec(o('temp_schedule', merged.get('temp_schedule'))),
        'kd_scheme': o('kd_scheme', merged.get('kd_scheme')),
        'kl_scale': o('kl_scale', merged.get('kl_scale')),
        'eval_ctx_lens': parse_eval_ctx_lens(o('eval_ctx_lens', merged.get('eval_ctx_lens'))),
        'reset_optim_on_ctx_change': bool(o('reset_optim_on_ctx_change', merged.get('reset_optim_on_ctx_change', False))),
        'grad_accum': o('grad_accum', merged.get('grad_accum')),
        'rope_scale': o('rope_scale', merged.get('rope_scale')),
        'save_best_on': o('save_best_on', merged.get('save_best_on', 'val_ppl_student')),
        'save_every_steps': o('save_every_steps', merged.get('save_every_steps', 5000)),
        'schedule_from_zero': bool(o('schedule_from_zero', merged.get('schedule_from_zero', False))),
        'reset_lrsched_on_resume': bool(o('reset_lrsched_on_resume', merged.get('reset_lrsched_on_resume', False))),
    }
    if cfg['alpha'] is not None:
        cfg['alpha'] = float(cfg['alpha'])
    if cfg['T'] is not None:
        cfg['T'] = float(cfg['T'])
    if cfg['grad_accum'] is not None:
        cfg['grad_accum'] = int(cfg['grad_accum'])
    if cfg['lr_peak'] is not None:
        cfg['lr_peak'] = float(cfg['lr_peak'])
    if cfg['rope_scale'] is not None:
        cfg['rope_scale'] = float(cfg['rope_scale'])
    if cfg['save_every_steps'] is not None:
        cfg['save_every_steps'] = int(cfg['save_every_steps'])
    return cfg
