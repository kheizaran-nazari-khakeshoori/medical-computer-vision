"""Image cache for faster preprocessing."""

from functools import lru_cache

from src.preprocessing import load_image, preprocess_image


@lru_cache(maxsize=128)
def cached_preprocess(path: str, train: bool = False):
    img = load_image(path)
    return preprocess_image(img, train=train)
