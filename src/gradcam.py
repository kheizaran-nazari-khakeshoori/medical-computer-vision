"""Grad-CAM for explainable AI heatmaps."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM for ResNet-style models."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output

        def backward_hook(_, grad_output, __):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Generate heatmap for input_tensor (1, C, H, W). Returns HxW numpy in [0,1]."""
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients[0].detach().cpu()
        acts = self.activations[0].detach().cpu()
        weights = grads.mean(dim=(1, 2))
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam = cam.numpy()
        # resize to input size
        h, w = input_tensor.shape[2:]
        cam = cv2.resize(cam, (w, h))
        return cam

    @staticmethod
    def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Overlay heatmap on RGB image (uint8)."""
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlayed = np.uint8(alpha * heatmap_color + (1 - alpha) * image)
        return overlayed
