"""Protecting against adversarial attacks."""
import torch
def fgsm_attack(image, epsilon, data_grad):
    sign = data_grad.sign()
    return torch.clamp(image + epsilon * sign, 0, 1)
def detect_adversarial(image, threshold=0.1):
    # simple stub: check noise level
    return False
