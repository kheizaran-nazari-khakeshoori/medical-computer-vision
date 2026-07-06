"""Image validation for uploaded medical scans."""

from pathlib import Path

import pydicom
from PIL import Image

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm", ".dicom"}
MAX_FILE_SIZE_MB = 20


def validate_image(path: str) -> tuple[bool, str]:
    p = Path(path)
    if not p.exists():
        return False, "file does not exist"
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False, f"unsupported format {p.suffix}, allowed: {ALLOWED_EXTENSIONS}"
    if p.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"file too large > {MAX_FILE_SIZE_MB}MB"
    # try to open
    try:
        if p.suffix.lower() in {".dcm", ".dicom"}:
            pydicom.dcmread(str(p))
        else:
            Image.open(p).verify()
        return True, "valid"
    except Exception as e:
        return False, f"corrupted image: {e}"


def validate_pil_image(image: Image.Image) -> tuple[bool, str]:
    if image.size[0] < 50 or image.size[1] < 50:
        return False, "image too small, minimum 50x50 required"
    if image.mode not in {"RGB", "L", "RGBA"}:
        return False, f"unsupported image mode {image.mode}"
    return True, "valid"
