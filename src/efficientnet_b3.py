"""Introducing efficientnet b3 variant for better accuracy."""

import torch.nn as nn
from torchvision import models


def get_efficientnet_b3(num_classes=2, pretrained=True):
    w = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_b3(weights=w)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m
