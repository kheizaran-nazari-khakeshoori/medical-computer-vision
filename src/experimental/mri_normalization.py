"""MRI normalization for brain scans."""

import cv2
import numpy as np


def z_score_normalize(image: np.ndarray) -> np.ndarray:
    mean = image.mean()
    std = image.std() or 1.0
    return (image - mean) / std


def min_max_normalize(image: np.ndarray) -> np.ndarray:
    min_val = image.min()
    max_val = image.max()
    if max_val == min_val:
        return np.zeros_like(image)
    return (image - min_val) / (max_val - min_val)


def clahe_enhance(image: np.ndarray) -> np.ndarray:
    """Enhance MRI contrast with CLAHE."""
    if image.dtype != np.uint8:
        image = (min_max_normalize(image) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)
