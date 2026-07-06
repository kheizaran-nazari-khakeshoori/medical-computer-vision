"""DICOM dataset loader for hospital data."""

from pathlib import Path

from torch.utils.data import Dataset

from src.preprocessing import get_preprocess_transform, load_image


class DicomDataset(Dataset):
    """Loads DICOM files from nested hospital folders."""

    def __init__(self, root_dir: str, train: bool = False):
        self.root = Path(root_dir)
        self.files = list(self.root.rglob("*.dcm")) + list(self.root.rglob("*.dicom"))
        self.transform = get_preprocess_transform(train=train)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = load_image(str(path))
        if self.transform:
            image = self.transform(image)
        # label inferred from parent folder name if available
        label = 0
        if "pneumonia" in str(path).lower() or "tumor" in str(path).lower():
            label = 1
        return image, label
