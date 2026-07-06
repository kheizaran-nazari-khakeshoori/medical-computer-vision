"""Implementing focal loss with label smoothing."""

import torch.nn as nn
import torch.nn.functional as F


class FocalLabelSmoothLoss(nn.Module):
    def __init__(self, gamma=2, smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        logp = F.log_softmax(inputs, dim=1)
        n_classes = inputs.size(1)
        smooth_target = (
            F.one_hot(targets, n_classes).float() * (1 - self.smoothing)
            + self.smoothing / n_classes
        )
        ce = -(smooth_target * logp).sum(dim=1)
        pt = (-ce).exp()
        return ((1 - pt) ** self.gamma * ce).mean()
