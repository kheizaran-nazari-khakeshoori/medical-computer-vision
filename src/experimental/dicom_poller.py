"""Needing dicom server polling for new scans."""

import time
from pathlib import Path


def poll_dicom_folder(folder: str, interval=30, callback=None):
    seen = set()
    folder = Path(folder)
    while True:
        for f in folder.glob("*.dcm"):
            if str(f) not in seen:
                seen.add(str(f))
                if callback:
                    callback(str(f))
        time.sleep(interval)
        break
