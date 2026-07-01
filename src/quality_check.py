"""Checking image quality before inference."""

import cv2
import numpy as np
from PIL import Image


def check_blur(image: np.ndarray | Image.Image, threshold=100):
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score > threshold, float(score)


def check_contrast(image: np.ndarray | Image.Image, threshold=20):
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray.std() > threshold, float(gray.std())
