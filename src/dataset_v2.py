"""Refactoring dataset class for better extensibility."""
from pathlib import Path
from torch.utils.data import Dataset
from src.preprocessing import load_image, get_preprocess_transform
class BaseMedicalDataset(Dataset):
    def __init__(self, root_dir: str, extensions=None, train=False):
        self.root = Path(root_dir)
        self.extensions = extensions or {".jpg",".png",".dcm"}
        self.files = [p for p in self.root.rglob("*") if p.suffix.lower() in self.extensions]
        self.transform = get_preprocess_transform(train=train)
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = load_image(str(self.files[idx]))
        if self.transform: img = self.transform(img)
        return img, 0
