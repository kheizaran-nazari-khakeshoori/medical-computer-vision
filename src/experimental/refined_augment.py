"""Refining data augmentation for rare diseases."""
import torchvision.transforms as T
from src.config import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD
def refined_augment():
    return T.Compose([T.RandomResizedCrop(IMAGE_SIZE, scale=(0.7,1.0)), T.RandomHorizontalFlip(0.5), T.ColorJitter(0.2,0.2), T.ToTensor(), T.Normalize(IMAGE_MEAN, IMAGE_STD)])
