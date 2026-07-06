"""Batch prediction for multiple patient scans."""

from pathlib import Path

from PIL import Image

from src.inference import predict
from src.preprocessing import load_image


def batch_predict(folder: str, model=None):
    folder = Path(folder)
    results = []
    for p in folder.rglob("*"):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm", ".dicom"}:
            continue
        try:
            image = (
                load_image(str(p))
                if p.suffix.lower() in {".dcm", ".dicom"}
                else Image.open(p).convert("RGB")
            )
            res = predict(image, model=model)
            res["file"] = str(p)
            results.append(res)
        except Exception as e:
            results.append({"file": str(p), "error": str(e)})
    return results
