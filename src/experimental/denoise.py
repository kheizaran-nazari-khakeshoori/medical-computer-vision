"""Noise reduction for medical images."""
import cv2
import numpy as np
def denoise_image(image: np.ndarray, h=10):
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
def median_denoise(image: np.ndarray, ksize=3):
    return cv2.medianBlur(image, ksize)
