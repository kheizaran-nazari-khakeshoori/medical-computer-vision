"""Unit tests for preprocessing pipeline."""

from PIL import Image
from src.preprocessing import preprocess_image, denormalize


def test_preprocess_shape():
    img = Image.new("RGB", (300, 300), color="gray")
    t = preprocess_image(img)
    assert t.shape == (3, 224, 224)


def test_denormalize_range():
    img = Image.new("RGB", (224, 224), color="white")
    t = preprocess_image(img)
    d = denormalize(t)
    assert d.min() >= 0 and d.max() <= 1.5
