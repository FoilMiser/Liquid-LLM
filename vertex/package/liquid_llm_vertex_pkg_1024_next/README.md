# Liquid LLM Vertex Package 1024 Next

Utility scripts for running knowledge distillation experiments, evaluation, and
pipeline orchestration for Liquid LLM on Vertex AI.

Arguments:
--resume_gcs_uri=gs://liquid-llm-bucket-2/stage0/checkpoints/vertex_runs/20251009-022648/best.pt
--block_size=1024
--teacher_name=gpt2-xl
--teacher_precision=fp16
--dataset_name=wikitext
--dataset_config=wikitext-103-raw-v1
--output_gcs_uri=gs://liquid-llm-bucket-2/stage0/checkpoints/vertex_runs/$(date +%Y%m%d-%H%M%S)
--eval_ctx_lens=512,1024
--train_steps=1000000
--warmup_steps=400
--lr_base=3.0e-4
--lr_peak=2.2e-4
--lr_scheduler=cosine
--grad_accum=1
--alpha=0.20
--T=1.5
--alpha_schedule=cosine:hold_steps=2000,start=0.20,end=0.05,total_steps=25000
--temp_schedule=linear:start=1.5,end=1.0,total_steps=10000
--kd_scheme=forward_kl
--kl_scale=T^2
--phase_base=24500
--phase_index=1
--eval_every=500
--fallback_save_every=1000
--save_every_steps=5000
--save_best_on=val_loss_student@1024
--always_save_latest=1
--log_train_ppl=1
--selftest=1
--schedule_from_zero
--reset_lrsched_on_resume
--hf_secret_name=hf_token
--gcs_root=gs://liquid-llm-bucket-2/stage0
--pipeline_mode=1
--improve_thresh=0.03
