"""Updating training loop with gradient clipping."""
import torch.nn as nn
def clip_gradients(model, max_norm=1.0):
    return nn.utils.clip_grad_norm_(model.parameters(), max_norm)
