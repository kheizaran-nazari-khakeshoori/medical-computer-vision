"""Custom exceptions for medical image handling."""


class InvalidImageError(Exception):
    """Raised when image is invalid or corrupted."""


class UnsupportedFormatError(Exception):
    """Raised when file format is not supported."""


class ModelNotLoadedError(Exception):
    """Raised when model weights are missing."""


class PredictionError(Exception):
    """Raised when inference fails."""
