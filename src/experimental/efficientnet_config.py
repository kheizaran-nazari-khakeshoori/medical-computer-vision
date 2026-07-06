"""Model config to support efficientnet."""

from src.config import NUM_CLASSES

EFFICIENTNET_VARIANTS = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b3"]


def get_efficientnet_config(variant="efficientnet_b0"):
    return {"model_name": variant, "num_classes": NUM_CLASSES, "pretrained": True, "dropout": 0.2}
