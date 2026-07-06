"""Backup utility for model weights."""

import shutil
from datetime import datetime
from pathlib import Path


def backup_model(src: str = "models/resnet50_medical.pth", dest_dir: str = "models/backups"):
    src_path = Path(src)
    if not src_path.exists():
        return None
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    backup_name = f"{src_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{src_path.suffix}"
    out = dest / backup_name
    shutil.copy(src_path, out)
    return str(out)
