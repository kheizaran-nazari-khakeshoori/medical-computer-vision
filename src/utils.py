"""Utility helpers: device, reproducibility, validation."""

import os
import random
from pathlib import Path

import numpy as np
import torch


def get_device() -> torch.device:
    """Return best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def is_valid_image_path(path: str | os.PathLike) -> bool:
    """Check if path exists and has a supported image extension."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm", ".dicom"}
    p = Path(path)
    return p.is_file() and p.suffix.lower() in valid_exts


def ensure_dir(path: str | os.PathLike) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
