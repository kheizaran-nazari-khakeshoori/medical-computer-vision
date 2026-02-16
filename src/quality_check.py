"""Checking image quality before inference."""
import cv2
import numpy as np
def check_blur(image: np.ndarray, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score > threshold, score
def check_contrast(image: np.ndarray, threshold=20):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray.std() > threshold
