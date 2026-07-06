"""Assisting clinicians with attention maps."""

import torch


def get_attention_weights(model, x):
    # placeholder for attention extraction
    with torch.no_grad():
        _ = model(x)
    return {}
