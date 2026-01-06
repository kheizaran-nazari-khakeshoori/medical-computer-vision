"""Image loading and preprocessing for X-Ray/MRI (PIL + DICOM)."""

import cv2
import numpy as np
import pydicom
from PIL import Image
import torch
from torchvision import transforms

from src.config import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD


def _apply_dicom_windowing(pixel_array: np.ndarray, ds) -> np.ndarray:
    """Apply DICOM windowing if metadata present, else normalize to 0-255."""
    # Use WindowCenter / WindowWidth if available
    try:
        center = ds.WindowCenter
        width = ds.WindowWidth
        # Can be MultiValue
        if isinstance(center, pydicom.multival.MultiValue):
            center = float(center[0])
        if isinstance(width, pydicom.multival.MultiValue):
            width = float(width[0])
        center, width = float(center), float(width)
        lower = center - width / 2
        upper = center + width / 2
        pixel_array = np.clip(pixel_array, lower, upper)
        pixel_array = ((pixel_array - lower) / (upper - lower) * 255.0)
        return pixel_array.astype(np.uint8)
    except Exception:
        # Fallback: min-max normalize
        pixel_array = pixel_array.astype(np.float32)
        pixel_array -= pixel_array.min()
        if pixel_array.max() > 0:
            pixel_array = pixel_array / pixel_array.max() * 255.0
        return pixel_array.astype(np.uint8)


def load_image(path: str) -> Image.Image:
    """
    Load image from path. Supports .dcm/.dicom via pydicom and standard
    formats via PIL. Always returns RGB PIL Image.
    """
    path_lower = path.lower()
    if path_lower.endswith((".dcm", ".dicom")):
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array

        # Handle photometric interpretation if needed (MONOCHROME1 inversion)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = np.max(arr) - arr

        # Apply rescale slope/intercept if present
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        if slope != 1 or intercept != 0:
            arr = arr.astype(np.float32) * slope + intercept

        arr = _apply_dicom_windowing(arr, ds)

        # DICOM is grayscale -> convert to RGB
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 1:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(arr)

    # Standard image
    img = Image.open(path).convert("RGB")
    return img


# Transforms for pretrained models (ResNet50/EfficientNet)
_preprocess_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
])

# Light augmentation for training
_train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
])


def preprocess_image(image: Image.Image, train: bool = False) -> torch.Tensor:
    """
    Preprocess PIL image to normalized tensor (C, H, W).
    Set train=True to apply augmentation.
    """
    if train:
        return _train_transform(image)
    return _preprocess_transform(image)


def preprocess_path(path: str, train: bool = False) -> torch.Tensor:
    """Convenience: load from path and preprocess."""
    img = load_image(path)
    return preprocess_image(img, train=train)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize tensor for visualization (reverses ImageNet normalization)."""
    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(IMAGE_STD).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


def get_preprocess_transform(train: bool = False):
    """Return torchvision transform object for use in Dataset."""
    return _train_transform if train else _preprocess_transform
