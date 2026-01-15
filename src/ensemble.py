"""Model ensemble for robust predictions."""

import torch
import torch.nn.functional as F


def ensemble_predict(models, image_tensor: torch.Tensor, weights=None):
    """Average predictions from multiple models."""
    if weights is None:
        weights = [1 / len(models)] * len(models)
    probs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0))
            probs.append(F.softmax(logits, dim=1)[0])
    # weighted average
    ensemble_prob = sum(w * p for w, p in zip(weights, probs))
    return ensemble_prob
