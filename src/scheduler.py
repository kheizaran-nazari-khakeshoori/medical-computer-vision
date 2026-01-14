"""Learning rate scheduler for optimizer."""

import torch


def get_scheduler(optimizer, name: str = "cosine", epochs: int = 10):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    return None
