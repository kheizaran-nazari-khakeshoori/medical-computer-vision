"""Grad-CAM plus plus for better localization."""
import torch
import cv2
import numpy as np
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.grads = None
        self.acts = None
        target_layer.register_forward_hook(lambda _, __, o: setattr(self, "acts", o))
        target_layer.register_full_backward_hook(lambda _, go, __: setattr(self, "grads", go[0]))
    def generate(self, x, class_idx=None):
        self.model.eval()
        out = self.model(x)
        idx = class_idx or out.argmax(1).item()
        self.model.zero_grad()
        out[0, idx].backward()
        grads, acts = self.grads[0].cpu(), self.acts[0].cpu()
        weights = grads.mean((1,2))
        cam = (weights[:,None,None] * acts).sum(0).numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cv2.resize(cam, (x.shape[3], x.shape[2]))
