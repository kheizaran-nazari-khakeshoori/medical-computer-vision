"""Centralizing error handling for api failures."""

import logging
from functools import wraps

logger = logging.getLogger(__name__)


def handle_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"error in {func.__name__}: {e}")
            return {"error": str(e), "success": False}

    return wrapper
