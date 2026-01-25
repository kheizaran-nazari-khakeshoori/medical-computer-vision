"""Guided backprop for visualization."""
import torch
def guided_backprop(model, image_tensor, target_class=None):
    model.eval()
    image_tensor.requires_grad = True
    output = model(image_tensor)
    target = target_class or output.argmax(1).item()
    model.zero_grad()
    output[0, target].backward()
    grad = image_tensor.grad[0].cpu().numpy()
    return grad.transpose(1,2,0)
