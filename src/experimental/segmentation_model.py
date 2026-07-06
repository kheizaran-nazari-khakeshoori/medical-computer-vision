"""Extending model to support segmentation."""

import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50


def get_segmentation_model(num_classes=2):
    m = fcn_resnet50(weights=None)
    m.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    return m
