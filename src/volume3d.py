"""Needing support for 3d mri volumes."""
import numpy as np
def load_volume(paths):
    import pydicom
    vols = [pydicom.dcmread(p).pixel_array for p in paths]
    return np.stack(vols, axis=0)
def volume_stats(volume):
    return {"mean": float(volume.mean()), "std": float(volume.std()), "shape": volume.shape}
