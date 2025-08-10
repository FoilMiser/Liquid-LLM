
# src/tokenization.py
from pathlib import Path
import sentencepiece as spm

def train_sentencepiece(input_glob, model_prefix, vocab_size=32000):
    spm.SentencePieceTrainer.Train(
        input=input_glob,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,
        input_sentence_size=1_000_000,
        shuffle_input_sentence=True
    )
