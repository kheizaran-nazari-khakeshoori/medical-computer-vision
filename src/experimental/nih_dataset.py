"""Introducing nih chest xray dataset loader."""

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.preprocessing import get_preprocess_transform


class NIHChestXray(Dataset):
    def __init__(self, csv_path, img_dir, train=False):
        self.img_dir = Path(img_dir)
        self.samples = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                self.samples.append((row["Image Index"], row["Finding Labels"]))
        self.transform = get_preprocess_transform(train=train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, _ = self.samples[idx]
        img = Image.open(self.img_dir / name).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0
