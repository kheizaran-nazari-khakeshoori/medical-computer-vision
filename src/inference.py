"""Prediction pipeline with confidence scores."""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src.config import CLASS_NAMES, MODEL_PATH
from src.model import get_model, load_model
from src.preprocessing import preprocess_image
from src.utils import get_device


def _load_default_model(device):
    """Load fine-tuned checkpoint if exists, else fallback to ImageNet."""
    ckpt = Path(MODEL_PATH)
    if ckpt.exists():
        try:
            return load_model(str(ckpt), device=str(device))
        except Exception:
            pass
    model = get_model(pretrained=True)
    model.to(device)
    return model


def predict(
    image: Image.Image, model: torch.nn.Module | None = None, device: torch.device | None = None
):
    """Run inference and return class, confidence and all probabilities."""
    device = device or get_device()
    if model is None:
        model = _load_default_model(device)
    else:
        model.to(device)
    model.eval()

    tensor = preprocess_image(image).unsqueeze(0).to(device)
    # optional quality gate
    try:
        from src.quality_check import check_blur

        ok, score = check_blur(image)
        if not ok:
            # still predict but could log low quality
            pass
    except Exception:
        pass
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        conf, idx = probs.max(dim=0)
        label = CLASS_NAMES[idx.item()] if idx.item() < len(CLASS_NAMES) else str(idx.item())
        return {
            "label": label,
            "confidence": float(conf.item()),
            "class_idx": int(idx.item()),
            "probabilities": {
                CLASS_NAMES[i]: float(probs[i].item())
                for i in range(len(probs))
                if i < len(CLASS_NAMES)
            },
        }


def predict_with_heatmap(image: Image.Image, model: torch.nn.Module | None = None):
    """Convenience wrapper that also generates Grad-CAM heatmap if model is ResNet50."""

    from src.gradcam import GradCAM

    device = get_device()
    if model is None:
        model = _load_default_model(device)
    else:
        model.to(device)
    model.eval()

    result = predict(image, model=model, device=device)

    # try Grad-CAM on last conv layer
    try:
        target_layer = model.layer4[-1]  # ResNet50
        cam = GradCAM(model, target_layer)
        tensor = preprocess_image(image).unsqueeze(0).to(device)
        tensor.requires_grad = True
        heatmap = cam.generate(tensor, class_idx=result["class_idx"])
        result["heatmap"] = heatmap
    except Exception:
        result["heatmap"] = None

    return result
