"""Fixing memory leak in inference."""
import torch
import gc
def inference_no_leak(model, tensor):
    model.eval()
    with torch.no_grad():
        out = model(tensor)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return out
