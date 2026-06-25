"""Segmentation mask generation for lesions."""
import cv2
import numpy as np
def generate_mask(image: np.ndarray, threshold=127):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask
def apply_mask(image: np.ndarray, mask: np.ndarray):
    return cv2.bitwise_and(image, image, mask=mask)
