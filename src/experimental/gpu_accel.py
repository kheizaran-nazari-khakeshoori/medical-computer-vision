"""Activating gpu acceleration for inference."""
import torch
def to_device(model, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)
def enable_cudnn_benchmark():
    torch.backends.cudnn.benchmark = True
