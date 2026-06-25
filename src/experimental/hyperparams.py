"""Hyperparameters config for training."""

from dataclasses import dataclass


@dataclass
class HyperParams:
    learning_rate: float = 1e-4
    batch_size: int = 16
    epochs: int = 10
    weight_decay: float = 1e-5
    dropout: float = 0.3
    optimizer: str = "adam"
    scheduler: str = "cosine"


DEFAULT_HPARAMS = HyperParams()
