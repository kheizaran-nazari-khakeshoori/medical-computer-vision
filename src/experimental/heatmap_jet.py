"""Enhancing heatmap visualization with jet colormap."""
import cv2
import numpy as np
from PIL import Image
def apply_jet_colormap(heatmap: np.ndarray) -> Image.Image:
    norm = (heatmap - heatmap.min()) / (heatmap.max() + 1e-8)
    color = cv2.applyColorMap(np.uint8(255*norm), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return Image.fromarray(color)
def blend_heatmap(image: Image.Image, heatmap: np.ndarray, alpha=0.4):
    img_arr = np.array(image.convert("RGB"))
    heat_color = np.array(apply_jet_colormap(heatmap).resize(image.size))
    blended = np.uint8(alpha*heat_color + (1-alpha)*img_arr)
    return Image.fromarray(blended)
