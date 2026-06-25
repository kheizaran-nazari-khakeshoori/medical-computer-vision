"""Protecting api with rate limiting."""
import time
from collections import defaultdict
_hits = defaultdict(list)
def is_rate_limited(key: str, limit=10, window=60):
    now = time.time()
    _hits[key] = [t for t in _hits[key] if now - t < window]
    if len(_hits[key]) >= limit:
        return True
    _hits[key].append(now)
    return False
