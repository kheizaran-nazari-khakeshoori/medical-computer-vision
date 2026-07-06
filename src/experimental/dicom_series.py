"""Supporting dicom series loading for 3d scans."""

from pathlib import Path

import numpy as np
import pydicom


def load_dicom_series(folder: str):
    files = sorted(Path(folder).glob("*.dcm"))
    slices = []
    for f in files:
        ds = pydicom.dcmread(str(f))
        slices.append(ds.pixel_array)
    volume = np.stack(slices, axis=0)
    return volume


def series_to_rgb_slices(volume):
    return [(volume[i] / volume.max() * 255).astype(np.uint8) for i in range(volume.shape[0])]
