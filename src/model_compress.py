"""Compressing model for edge deployment."""
import torch
def quantize_model(model):
    q = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return q
def prune_model(model, amount=0.3):
    import torch.nn.utils.prune as prune
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            prune.l1_unstructured(m, name="weight", amount=amount)
    return model
