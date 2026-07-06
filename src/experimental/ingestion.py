"""Data ingestion script for new datasets."""

import shutil
from pathlib import Path


def ingest_dataset(source: str, dest: str = "data/raw"):
    src = Path(source)
    dst = Path(dest)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.rglob("*"):
        if f.suffix.lower() in {".jpg", ".png", ".dcm"}:
            shutil.copy(f, dst / f.name)
    return str(dst)
