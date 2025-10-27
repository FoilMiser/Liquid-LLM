"""Student model loading and precision helpers."""

import logging
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from . import gcs_io

LOGGER = logging.getLogger(__name__)


def determine_dtype(precision: str) -> torch.dtype:
    if precision == "float16":
        return torch.float16
    return torch.bfloat16


def get_autocast(dtype: torch.dtype):
    target_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16
    return torch.cuda.amp.autocast(dtype=target_dtype)


def load_student_model(
    resume_gcs_uri: str,
    device: torch.device,
    precision: str = "bfloat16",
    base_model_id: str = "meta-llama/Llama-3.1-8B",
    width_scale: float = 1.2,
    cache_dir: str = "/cache/student",
) -> Tuple[torch.nn.Module, torch.dtype]:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    local_ckpt = Path(cache_dir) / "stage1.pt"
    LOGGER.info("Downloading student checkpoint from %s", resume_gcs_uri)
    gcs_io.download_to_path(resume_gcs_uri, local_ckpt)
    LOGGER.info("Loading student base config %s", base_model_id)
    config = AutoConfig.from_pretrained(base_model_id)
    if hasattr(config, "intermediate_size"):
        config.intermediate_size = int(config.intermediate_size * width_scale)
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = config.num_hidden_layers + 1
    model = AutoModelForCausalLM.from_config(config)
    state_dict = torch.load(local_ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        LOGGER.info("Missing keys when loading student: %s", missing[:10])
    if unexpected:
        LOGGER.info("Unexpected keys when loading student: %s", unexpected[:10])
    dtype = determine_dtype(precision)
    model.to(device=device, dtype=dtype)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            try:
                module.gradient_checkpointing = True
            except Exception:  # pragma: no cover - best effort
                pass
    model.train()
    LOGGER.info("Student model ready on %s with dtype %s", device, dtype)
    return model, dtype


__all__ = ["load_student_model", "determine_dtype", "get_autocast"]
