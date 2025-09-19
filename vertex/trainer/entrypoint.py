# vertex/trainer/entrypoint.py
import os, io, sys, time, argparse, pathlib, re
from datetime import datetime
from typing import Optional, Tuple
from google.cloud import storage

# ---- You likely already use these in your notebook:
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- 1) IMPORT YOUR LIQUID STUDENT + TRAIN LOOP ----------
# Example (edit to match your repo):
# from liquid_llm.models.liquid import build_student_model
# from liquid_llm.training.stage0 import run_training
# ---------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_name", default="gpt2-xl")
    p.add_argument("--dataset_name", default="wikitext")
    p.add_argument("--dataset_config", default="wikitext-103-raw-v1")
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--alpha_kd", type=float, default=0.9)  # CE weight
    p.add_argument("--temp_kd", type=float, default=4.0)
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--resume_gcs_uri", default="", help="gs://bucket/path/to/checkpoints or specific file")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# ----------------- GCS helpers -----------------
def split_gcs_uri(uri: str) -> Tuple[str, str]:
    assert uri.startswith("gs://"), f"Not a GCS URI: {uri}"
    no = uri[5:]
    bucket, _, prefix = no.partition("/")
    return bucket, prefix

def gcs_latest_blob(gcs_dir: str, pattern=r".*\.(pt|pth)$") -> Optional[str]:
    bucket, prefix = split_gcs_uri(gcs_dir)
    client = storage.Client()
    blobs = list(client.list_blobs(bucket, prefix=prefix))
    if not blobs:
        return None
    rgx = re.compile(pattern)
    blobs = [b for b in blobs if rgx.match(b.name)]
    if not blobs:
        return None
    blobs.sort(key=lambda b: b.updated, reverse=True)
    return f"gs://{bucket}/{blobs[0].name}"

def gcs_download_to(gcs_uri: str, local_path: str):
    bucket, prefix = split_gcs_uri(gcs_uri)
    client = storage.Client()
    blob = client.bucket(bucket).blob(prefix)
    pathlib.Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    return local_path

def gcs_upload(local_path: str, gcs_dir: str):
    bucket, prefix = split_gcs_uri(gcs_dir)
    client = storage.Client()
    fname = pathlib.Path(local_path).name
    blob = client.bucket(bucket).blob(f"{prefix.rstrip('/')}/{fname}")
    blob.upload_from_filename(local_path)
    return f"gs://{bucket}/{blob.name}"

# ------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Vertex sets these when you pass a base_output_dir in the job:
    aip_ckpt_dir = os.getenv("AIP_CHECKPOINT_DIR", "").strip()   # gs://.../checkpoints/
    aip_log_dir  = os.getenv("AIP_TENSORBOARD_LOG_DIR", "").strip()
    base_ckpt_dir = args.resume_gcs_uri or aip_ckpt_dir

    # Teacher + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_name)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_name).eval().cuda()

    # Dataset (stream or download)
    ds = load_dataset(args.dataset_name, args.dataset_config)
    # You likely already have tokenization + blockifying in your notebook.
    # Keep your existing code. Only ensure block_size matches args.block_size.

    # ---------- 2) BUILD YOUR LIQUID STUDENT ----------
    # student = build_student_model(d_model=768, n_layers=10, ...)
    # student.cuda()
    # --------------------------------------------------

    # Resume (optional)
    resume_local = ""
    if base_ckpt_dir:
        if base_ckpt_dir.endswith(".pt") or base_ckpt_dir.endswith(".pth"):
            resume_local = gcs_download_to(base_ckpt_dir, f"/tmp/resume.pt")
        else:
            latest = gcs_latest_blob(base_ckpt_dir)
            if latest:
                resume_local = gcs_download_to(latest, f"/tmp/resume.pt")

    # ---------- 3) RUN YOUR EXISTING TRAIN LOOP ----------
    # IMPORTANT: Make your run_training(...) accept:
    # - teacher, tokenizer, dataset
    # - hyperparams (lr, alpha_kd, temp_kd, etc.)
    # - resume_from (str path to local ckpt) or None
    # - save_callback(step, local_ckpt_path)
    #
    # Example:
    #
    # def save_cb(step:int, local_path:str):
    #     if aip_ckpt_dir:
    #         gcs_uri = gcs_upload(local_path, aip_ckpt_dir)
    #         print(f"[vertex] uploaded checkpoint → {gcs_uri}")
    #
    # run_training(student=student,
    #              teacher=teacher,
    #              tokenizer=tokenizer,
    #              dataset=ds,
    #              block_size=args.block_size,
    #              batch_size=args.batch_size,
    #              grad_accum=args.grad_accum,
    #              lr=args.lr,
    #              alpha_kd=args.alpha_kd,
    #              temp_kd=args.temp_kd,
    #              max_steps=args.max_steps,
    #              eval_every=args.eval_every,
    #              save_every=args.save_every,
    #              resume_from=resume_local or None,
    #              log_dir=aip_log_dir or None,
    #              save_callback=save_cb)
    #
    # -------------------------------------------------------

    print("This is a template. Wire in your run_training(...) and student.")
    print("See comments in entrypoint.py for exact places to plug in.")

if __name__ == "__main__":
    main()