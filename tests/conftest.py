"""Organizing tests with fixtures."""
import pytest
from PIL import Image
@pytest.fixture
def dummy_image():
    return Image.new("RGB", (224,224), color="gray")
@pytest.fixture
def dummy_tensor(dummy_image):
    from src.preprocessing import preprocess_image
    return preprocess_image(dummy_image)
