"""Edge detection for tumor boundaries."""
import cv2
import numpy as np
def detect_edges(image: np.ndarray, low=50, high=150):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, low, high)
def find_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
