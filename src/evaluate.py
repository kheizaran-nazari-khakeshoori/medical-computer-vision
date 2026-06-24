"""Evaluation metrics for model performance."""

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.dataset import MedicalImageDataset
from src.utils import get_device


def evaluate(model, data_dir: str = "data", batch_size: int = 16):
    from src.config import CLASS_NAMES

    device = get_device()
    model.to(device)
    model.eval()
    dataset = MedicalImageDataset(data_dir, train=False)
    if len(dataset) == 0:
        print(f"no images found in {data_dir}")
        return {}
    loader = DataLoader(dataset, batch_size=batch_size)
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    acc = accuracy_score(y_true, y_pred)
    # ensure labels align with CLASS_NAMES
    labels = list(range(len(CLASS_NAMES)))
    print(f"accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES, zero_division=0))
    print("confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=labels))
    return {"accuracy": acc, "y_true": y_true, "y_pred": y_pred}
