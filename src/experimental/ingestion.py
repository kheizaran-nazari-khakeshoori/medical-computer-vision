"""Data ingestion script for new datasets."""
from pathlib import Path
import shutil
def ingest_dataset(source: str, dest: str = "data/raw"):
    src = Path(source)
    dst = Path(dest)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.rglob("*"):
        if f.suffix.lower() in {".jpg",".png",".dcm"}:
            shutil.copy(f, dst / f.name)
    return str(dst)
