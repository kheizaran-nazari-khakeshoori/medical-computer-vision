"""Improving contrast enhancement for low dose scans."""

import cv2
import numpy as np


def enhance_lowdose(image: np.ndarray, alpha=1.5, beta=10):
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(enhanced.shape) == 3:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return clahe.apply(enhanced)
