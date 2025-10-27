"""Teacher model utilities."""

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import load_manifest, resolve_paths, iter_jsonl

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    import gcsfs


def load_teacher(
    model_id: str = "meta-llama/Llama-3.2-3B",
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: str = "/cache/hf",
):
    LOGGER.info("Loading teacher model %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, cache_dir=cache_dir)
    model.to("cuda")
    model.eval()
    return model, tokenizer


def load_teacher_tokenizer(
    model_id: str = "meta-llama/Llama-3.2-3B",
    cache_dir: str = "/cache/hf",
):
    LOGGER.info("Loading teacher tokenizer %s", model_id)
    return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)


def precompute_teacher_logits(
    manifest_uri: str,
    output_dir: str = "gs://liquid-llm-bucket-2/teacher/llama-3.2-3b/logits/",
    model_id: str = "meta-llama/Llama-3.2-3B",
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 4,
    seq_len: int = 1024,
) -> None:
    model, tokenizer = load_teacher(model_id=model_id, dtype=dtype)
    import gcsfs

    fs = gcsfs.GCSFileSystem()
    manifest = load_manifest(manifest_uri)
    shard_idx = 0
    for entry in manifest:
        paths = resolve_paths(fs, entry.path)
        for path in paths:
            LOGGER.info("Precomputing logits for %s", path)
            tokens: List[List[int]] = []
            for row in iter_jsonl(fs, path):
                text = row.get("text", "")
                tokens_ids = tokenizer.encode(text, add_special_tokens=False)
                tokens_ids = tokens_ids[: seq_len - 1]
                eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id
                pad_id = tokenizer.pad_token_id or eos_id
                tokens_ids.append(eos_id)
                if len(tokens_ids) < seq_len:
                    tokens_ids = tokens_ids + [pad_id] * (seq_len - len(tokens_ids))
                tokens.append(tokens_ids)
                if len(tokens) == batch_size:
                    _flush_logits(model, tokens, shard_idx, output_dir)
                    shard_idx += 1
                    tokens = []
            if tokens:
                _flush_logits(model, tokens, shard_idx, output_dir)
                shard_idx += 1


def _flush_logits(model, batch_tokens: List[List[int]], shard_idx: int, output_dir: str) -> None:
    import gcsfs

    fs = gcsfs.GCSFileSystem()
    input_ids = torch.tensor(batch_tokens, device="cuda")
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits.cpu()
    for seq_tokens, seq_logits in zip(batch_tokens, logits):
        key = hashlib.sha1(torch.tensor(seq_tokens, dtype=torch.int64).numpy().tobytes()).hexdigest()
        local_path = Path("/tmp") / f"{key}.pt"
        torch.save({"logits": seq_logits, "input_ids": torch.tensor(seq_tokens)}, local_path)
        target_uri = f"{output_dir.rstrip('/')}/{key}.pt"
        LOGGER.info("Writing teacher logits shard %s", target_uri)
        fs.put(str(local_path), target_uri)


__all__ = ["load_teacher", "load_teacher_tokenizer", "precompute_teacher_logits"]
