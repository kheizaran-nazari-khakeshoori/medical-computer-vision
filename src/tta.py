"""Test time augmentation for better accuracy."""

import torch
from torchvision import transforms
from PIL import Image


def tta_predict(model, image: Image.Image, n_augmentations: int = 5, device="cpu"):
    """Average predictions over augmentations."""
    from src.preprocessing import preprocess_image

    model.eval()
    augment = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
    ])
    probs = []
    for _ in range(n_augmentations):
        aug_img = augment(image)
        tensor = preprocess_image(aug_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs.append(torch.softmax(logits, dim=1)[0])
    return torch.stack(probs).mean(dim=0)
