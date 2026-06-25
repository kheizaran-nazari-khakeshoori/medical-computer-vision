"""Supporting multi-gpu training."""
import torch
def wrap_dataparallel(model):
    if torch.cuda.device_count() > 1:
        return torch.nn.DataParallel(model)
    return model
