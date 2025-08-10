
# Liquid-LLM

Liquid Neural Network (LNN) student model distilled from **DeepSeek-R1-0528**.  
Train on Google Colab Pro for heavy stages; iterate and evaluate locally (16GB RAM / 8GB NVIDIA GPU) with Conda + JupyterLab.

## Quickstart

### 1) Create Conda env (local)
```bash
conda create -n Liquid-LLM python=3.11 -y
conda activate Liquid-LLM
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install torchdiffeq transformers datasets accelerate evaluate sentencepiece bitsandbytes wandb huggingface_hub
pip install jupyterlab ipykernel
python -m ipykernel install --user --name Liquid-LLM --display-name "Liquid-LLM"
```

> If you have a different CUDA version, adjust the PyTorch index URL accordingly: https://pytorch.org/get-started/locally/

### 2) Configure Accelerate (optional, for multi-GPU/TPU/CPU configs)
```bash
accelerate config default
```

### 3) Hugging Face (optional, for syncing checkpoints)
```bash
huggingface-cli login
```

### 4) Run a tiny smoke test
```bash
python scripts/smoke_test.py
```

### 5) Colab
Open `notebooks/colab_train.ipynb` in Colab, install deps, and run Stage A (KD) on a larger shard.

## Repo layout
```
data/               # raw text shards and distilled datasets (documented; not committed)
models/             # checkpoints, tokenizer (not committed)
src/                # model + training code
scripts/            # helper scripts (teacher logits, self-instruct, smoke test)
notebooks/          # Colab / local notebooks
```

## Ethics & Licensing
- Use only permissive/open datasets (Wikipedia CC-BY-SA, StackExchange CC-BY-SA, Project Gutenberg public domain, etc.).
- Document sources in `DATA_CARD.md`.
- Code licensed Apache-2.0 (see LICENSE).
