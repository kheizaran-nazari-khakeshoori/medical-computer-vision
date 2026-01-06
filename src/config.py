"""Central configuration for medical computer vision project."""

# Image preprocessing
IMAGE_SIZE = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet stats for pretrained ResNet/EfficientNet
IMAGE_STD = [0.229, 0.224, 0.225]
NUM_CHANNELS = 3

# Classes (extend as needed: e.g., ["normal", "pneumonia", "covid"] or ["no_tumor", "tumor"])
CLASS_NAMES = ["normal", "diseased"]
NUM_CLASSES = len(CLASS_NAMES)

# Model
MODEL_NAME = "resnet50"  # options: resnet50, efficientnet_b0
PRETRAINED = True

# Paths
DATA_DIR = "data"
MODELS_DIR = "models"

# Training defaults
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
RANDOM_SEED = 42

# DICOM windowing defaults (used if no windowing metadata present)
DICOM_WINDOW_CENTER = 40
DICOM_WINDOW_WIDTH = 400
