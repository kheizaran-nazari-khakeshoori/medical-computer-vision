"""Dataset loader for medical images (X-Ray / MRI)."""

from pathlib import Path
from typing import Callable, List, Tuple

from PIL import Image
from torch.utils.data import Dataset

from src.config import CLASS_NAMES
from src.preprocessing import get_preprocess_transform, load_image


class MedicalImageDataset(Dataset):
    """Simple folder-based dataset: data/<class_name>/*.jpg|png|dcm"""

    def __init__(self, root_dir: str, train: bool = False, transform: Callable | None = None):
        self.root = Path(root_dir)
        self.train = train
        self.transform = transform or get_preprocess_transform(train=train)
        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx = {}

        if self.root.exists():
            discovered = [d.name for d in self.root.iterdir() if d.is_dir()]
            # keep canonical order from config to avoid label flip (e.g. normal=0, diseased=1)
            ordered = [c for c in CLASS_NAMES if c in discovered]
            # include any extra discovered classes not in config
            extras = sorted([c for c in discovered if c not in CLASS_NAMES])
            classes = ordered + extras
            self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
            for cls, idx in self.class_to_idx.items():
                for p in (self.root / cls).rglob("*"):
                    if p.suffix.lower() in {
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".tiff",
                        ".dcm",
                        ".dicom",
                    }:
                        self.samples.append((p, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = load_image(str(path))
        except Exception:
            # fallback to PIL directly
            image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_names(self) -> List[str]:
        return sorted(self.class_to_idx, key=lambda k: self.class_to_idx[k])
