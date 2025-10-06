from transformers import AutoTokenizer

def get_tokenizer(model_name: str, use_fast: bool = True):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
