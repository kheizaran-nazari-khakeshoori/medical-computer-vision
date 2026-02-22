"""Adding attention heatmap for model interpretability."""
import cv2
import numpy as np
def attention_heatmap(attn: np.ndarray, size=(224,224)):
    norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    heat = cv2.resize(norm, size)
    color = cv2.applyColorMap(np.uint8(255*heat), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
