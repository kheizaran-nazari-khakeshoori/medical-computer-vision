"""Extending augmentation with elastic transform."""
import numpy as np
import cv2
def elastic_transform(image: np.ndarray, alpha=34, sigma=4):
    shape = image.shape[:2]
    dx = cv2.GaussianBlur((np.random.rand(*shape)*2-1), (0,0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(*shape)*2-1), (0,0), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
