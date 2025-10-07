import torch

def build_optimizer(
    model,
    lr: float,
    weight_decay: float,
    betas=(0.9, 0.95),
    eps: float = 1e-8,
    fused: bool | None = True,
):
    """
    Create an AdamW optimizer with sane defaults.

    Args:
        model: nn.Module (the student model)
        lr: learning rate
        weight_decay: weight decay coefficient
        betas: Adam beta parameters
        eps: numerical stability term
        fused: Use fused AdamW if available (PyTorch 2.x on CUDA)

    Returns:
        torch.optim.AdamW
    """
    optim_kwargs = dict(
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    try:
        if fused and hasattr(torch.optim, "AdamW"):
            # PyTorch 2.x AdamW will automatically enable fused CUDA kernels
            return torch.optim.AdamW(model.parameters(), **optim_kwargs)
    except Exception:
        pass

    # Fallback: standard AdamW
    return torch.optim.AdamW(model.parameters(), **optim_kwargs)
