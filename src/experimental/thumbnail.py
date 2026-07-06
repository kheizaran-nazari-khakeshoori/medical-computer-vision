"""Thumbnail generator for scan previews."""

from PIL import Image


def generate_thumbnail(image: Image.Image, size=(128, 128)):
    thumb = image.copy()
    thumb.thumbnail(size, Image.LANCZOS)
    return thumb


def save_thumbnail(image: Image.Image, path: str, size=(128, 128)):
    thumb = generate_thumbnail(image, size)
    thumb.save(path)
    return path
