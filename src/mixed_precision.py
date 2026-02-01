"""Enabling mixed precision training for speed."""
import torch
from torch.cuda.amp import autocast, GradScaler
def train_with_amp(model, loader, optimizer, criterion, device):
    scaler = GradScaler()
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return loss.item()
