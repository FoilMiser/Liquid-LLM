# liquid_llm/data/wikitext.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import random
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


def _group_texts(examples: Dict[str, List[List[int]]], block_size: int) -> Dict[str, List[List[int]]]:
    # Concatenate then split into exact block_size chunks; no attention_mask here.
    concatenated: List[int] = [tok_id for seq in examples["input_ids"] for tok_id in seq]
    total_length = (len(concatenated) // block_size) * block_size
    if total_length == 0:
        return {"input_ids": [], "labels": []}
    concatenated = concatenated[:total_length]
    chunks = [concatenated[i : i + block_size] for i in range(0, total_length, block_size)]
    return {"input_ids": chunks, "labels": [c[:] for c in chunks]}


def build_dataloaders(
    dataset_name: str,
    dataset_config: str,
    teacher_name: str,
    block_size: int,
    global_batch: int = 128,
    micro_batch: int = 4,
    num_workers: int = 2,
    seed: Optional[int] = None,
    hf_token: Optional[str] = None,
    **_: dict,  # tolerate extra kwargs
) -> Tuple[DataLoader, DataLoader, int, int, AutoTokenizer]:
    # Seeding (no datasets.set_seed)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    raw: DatasetDict = load_dataset(dataset_name, dataset_config)

    auth_kwargs = {"token": hf_token} if hf_token else {}
    tok = AutoTokenizer.from_pretrained(teacher_name, use_fast=True, **auth_kwargs)
    if tok.pad_token is None:
        if tok.eos_token is None:
            tok.add_special_tokens({"eos_token": "<|endoftext|>"})
        tok.pad_token = tok.eos_token

    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        out = tok(
            batch["text"],
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            padding=False,
            truncation=False,
        )
        return {"input_ids": out["input_ids"]}

    tokenized = raw.map(
        tokenize,
        batched=True,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing",
    )

    lm = tokenized.map(
        lambda ex: _group_texts(ex, block_size),
        batched=True,
        desc=f"Grouping into blocks of {block_size}",
    )

    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=None)

    # Deterministic DataLoader shuffle
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    def _collate_causal(features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        # Next-token prediction: each position targets the subsequent token.
        # The final position has no target and is masked with -100 so it does
        # not contribute to the loss (mirrors HF causal LM convention).
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100

        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    def _to_loader(split: str, batch_size: int) -> DataLoader:
        ds = lm[split]
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_collate_causal,
            drop_last=True,
            generator=g,
        )

    train_loader = _to_loader("train", micro_batch)
    val_split = "validation" if "validation" in lm else ("test" if "test" in lm else "train")
    val_loader = _to_loader(val_split, micro_batch)

    vocab_size = len(tok)
    pad_id = tok.pad_token_id
    return train_loader, val_loader, vocab_size, pad_id, tok
