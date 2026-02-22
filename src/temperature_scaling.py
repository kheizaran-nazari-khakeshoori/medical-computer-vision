"""Improving model calibration with temperature scaling."""
import torch
import torch.nn as nn
class TemperatureScaling(nn.Module):
    def __init__(self): super().__init__(); self.T = nn.Parameter(torch.ones(1))
    def forward(self, logits): return logits / self.T
    def fit(self, logits, labels, lr=0.01, steps=100):
        opt = torch.optim.LBFGS([self.T], lr=lr)
        ce = nn.CrossEntropyLoss()
        def closure():
            opt.zero_grad(); loss = ce(self.forward(logits), labels); loss.backward(); return loss
        for _ in range(steps): opt.step(closure)
        return float(self.T.item())
