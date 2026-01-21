"""Image resize utility for preprocessing."""
import cv2
import numpy as np
from PIL import Image
def resize_image(image: Image.Image, size=(224,224)):
    return image.resize(size, Image.BILINEAR)
def resize_array(arr: np.ndarray, size=(224,224)):
    return cv2.resize(arr, size)
