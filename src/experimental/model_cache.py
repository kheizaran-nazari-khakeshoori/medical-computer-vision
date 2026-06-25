"""Caching model weights for faster startup."""
from functools import lru_cache
from src.model import get_model
@lru_cache(maxsize=2)
def get_cached_model(name="resnet50"):
    model = get_model()
    model.eval()
    return model
def clear_model_cache():
    get_cached_model.cache_clear()
