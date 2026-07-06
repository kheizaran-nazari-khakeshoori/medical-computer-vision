"""Data splitting utilities for train test sets."""

import random
import shutil
from pathlib import Path


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    random.seed(seed)
    source = Path(source_dir)
    output = Path(output_dir)
    for split in ["train", "val", "test"]:
        (output / split).mkdir(parents=True, exist_ok=True)

    for class_dir in source.iterdir():
        if not class_dir.is_dir():
            continue
        # skip output dir if inside source (e.g., data/processed inside data)
        if (
            class_dir.resolve() == output.resolve()
            or output.resolve() in class_dir.resolve().parents
        ):
            continue
        files = [p for p in class_dir.glob("*") if p.is_file()]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }
        for split, split_files in splits.items():
            dest = output / split / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy(f, dest / f.name)
    print(f"dataset split completed at {output_dir}")
