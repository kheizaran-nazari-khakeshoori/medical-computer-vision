"""Fixing grayscale conversion for mri images."""

import numpy as np
from PIL import Image


def fix_grayscale(image: Image.Image) -> Image.Image:
    if image.mode == "L":
        return image.convert("RGB")
    if image.mode == "RGBA":
        return image.convert("RGB")
    if image.mode == "P":
        return image.convert("RGB")
    return image


def ensure_three_channel(arr: np.ndarray):
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    return arr
