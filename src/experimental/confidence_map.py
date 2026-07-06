"""Confidence heatmap for prediction scores."""

import cv2
import numpy as np


def confidence_heatmap(scores: np.ndarray, shape=(224, 224)):
    norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    heat = cv2.resize(norm, shape)
    heat_color = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
