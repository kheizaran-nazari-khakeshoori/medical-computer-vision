"""Training loop for medical image classifier."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, RANDOM_SEED
from src.dataset import MedicalImageDataset
from src.model import get_model
from src.utils import get_device, set_seed


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


def train(data_dir: str = "data", epochs: int = NUM_EPOCHS, batch_size: int = BATCH_SIZE):
    set_seed(RANDOM_SEED)
    device = get_device()
    dataset = MedicalImageDataset(data_dir, train=True)
    if len(dataset) == 0:
        print(f"no images found in {data_dir}, skipping training")
        return
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"epoch {epoch+1}/{epochs} loss: {loss:.4f}")

    torch.save(model.state_dict(), "models/resnet50_medical.pth")
    print("saved model to models/resnet50_medical.pth")


if __name__ == "__main__":
    train()
