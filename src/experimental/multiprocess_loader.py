"""Accelerating image loading with multiprocessing."""

from multiprocessing import Pool

from src.preprocessing import load_image


def load_batch(paths, workers=4):
    with Pool(workers) as p:
        return p.map(load_image, paths)
