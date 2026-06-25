"""Checkpoint saving and loading for model weights."""

from pathlib import Path
import torch


def save_checkpoint(model, optimizer, epoch: int, path: str = "models/checkpoint.pth"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
    }, path)
    print(f"checkpoint saved to {path}")


def load_checkpoint(path: str, model, optimizer=None, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"loaded checkpoint from {path} epoch {ckpt.get('epoch', '?')}")
    return ckpt.get("epoch", 0)
