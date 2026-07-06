"""Optimizing inference pipeline for batch processing."""

import torch
from torch.utils.data import DataLoader

from src.utils import get_device


def batched_inference(model, dataset, batch_size=16, device=None):
    device = device or get_device()
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    results = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            results.extend(probs.cpu().tolist())
    return results
