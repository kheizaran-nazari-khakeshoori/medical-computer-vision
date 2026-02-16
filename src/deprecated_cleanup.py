"""Removing deprecated preprocessing functions."""
# removed old resize_fast - use src.resize.resize_image instead
DEPRECATED = ["resize_fast", "old_normalize"]
def check_deprecated(name: str):
    if name in DEPRECATED:
        raise DeprecationWarning(f"{name} is deprecated")
