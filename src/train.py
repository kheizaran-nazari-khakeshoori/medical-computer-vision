"""Training loop for medical image classifier."""

import os
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset

from src.config import BATCH_SIZE, LEARNING_RATE, MODELS_DIR, NUM_EPOCHS, RANDOM_SEED
from src.dataset import MedicalImageDataset
from src.model import get_model
from src.utils import ensure_dir, get_device, set_seed


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.cpu().tolist())
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    return total_loss / max(1, len(loader)), acc


def _stratified_split(dataset, val_ratio=0.2, seed=RANDOM_SEED):
    """Stratified split preserving class distribution."""
    from collections import defaultdict
    import random

    random.seed(seed)
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    train_idx, val_idx = [], []
    for label, indices in class_to_indices.items():
        random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    random.shuffle(train_idx)
    random.shuffle(val_idx)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def train(data_dir: str = "data", epochs: int = NUM_EPOCHS, batch_size: int = BATCH_SIZE):
    set_seed(RANDOM_SEED)
    device = get_device()
    dataset = MedicalImageDataset(data_dir, train=True)
    if len(dataset) == 0:
        print(f"no images found in {data_dir}, skipping training")
        return
    train_ds, val_ds = _stratified_split(dataset, val_ratio=0.2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    try:
        from src.scheduler import get_scheduler

        scheduler = get_scheduler(optimizer, name="cosine", epochs=epochs)
    except Exception:
        scheduler = None

    best_val_acc = 0.0
    ensure_dir(MODELS_DIR)
    save_path = os.path.join(MODELS_DIR, "resnet50_medical.pth")
    patience = 3
    stale = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"epoch {epoch+1}/{epochs} train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # save best checkpoint + early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved best model (val_acc {val_acc:.4f}) to {save_path}")
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"early stopping at epoch {epoch+1}")
                break

    # ensure final model saved even if no improvement
    if not Path(save_path).exists():
        torch.save(model.state_dict(), save_path)
    print(f"training done, best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
