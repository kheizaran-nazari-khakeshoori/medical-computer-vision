"""Heatmap overlay helpers for Grad-CAM visualization."""

import cv2
import numpy as np
from PIL import Image


def overlay_heatmap(pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Overlay heatmap (HxW in [0,1]) on PIL image."""
    image = np.array(pil_image.convert("RGB"))
    # resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(alpha * heatmap_color + (1 - alpha) * image)
    return Image.fromarray(overlay)


def create_side_by_side(original: Image.Image, overlay: Image.Image) -> Image.Image:
    """Create side-by-side comparison image."""
    w, h = original.size
    combined = Image.new("RGB", (w * 2, h))
    combined.paste(original, (0, 0))
    combined.paste(overlay, (w, 0))
    return combined
