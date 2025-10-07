from typing import Optional

from transformers import AutoTokenizer


def get_tokenizer(model_name: str, use_fast: bool = True, hf_token: Optional[str] = None):
    auth_kwargs = {"token": hf_token} if hf_token else {}
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast, **auth_kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
