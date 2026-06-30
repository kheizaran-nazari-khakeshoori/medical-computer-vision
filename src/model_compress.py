"""Compressing model for edge deployment."""

import torch


def quantize_model(model):
    """Dynamic quantization for Linear layers (CPU speedup)."""
    try:
        q = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        return q
    except Exception as e:
        print(f"quantization failed: {e}")
        return model


def prune_model(model, amount=0.3):
    """L1 unstructured pruning for Conv2d layers."""
    import torch.nn.utils.prune as prune

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            prune.l1_unstructured(m, name="weight", amount=amount)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
