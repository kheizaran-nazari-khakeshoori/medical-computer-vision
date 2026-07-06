"""ResNet50 classifier for medical image diagnosis."""

import torch
import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES


def get_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Create ResNet50 with custom classifier head."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # freeze early layers optionally - keep all trainable for fine-tuning
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_model(
    checkpoint_path: str, num_classes: int = NUM_CLASSES, device: str = "cpu"
) -> nn.Module:
    """Load model from checkpoint."""
    model = get_model(num_classes=num_classes, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    # handle checkpoints saved as dict with 'state_dict' key
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
