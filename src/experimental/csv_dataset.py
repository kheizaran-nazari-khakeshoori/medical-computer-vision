"""CSV dataset loader for annotations."""

import csv

from PIL import Image
from torch.utils.data import Dataset

from src.preprocessing import get_preprocess_transform


class CSVDataset(Dataset):
    def __init__(self, csv_path: str, train=False):
        self.samples = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["path"], int(row["label"])))
        self.transform = get_preprocess_transform(train=train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
