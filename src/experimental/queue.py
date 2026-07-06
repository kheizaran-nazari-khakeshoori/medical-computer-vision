"""Batch processing queue for large datasets."""

from collections import deque


class ProcessingQueue:
    def __init__(self):
        self.q = deque()

    def add(self, path: str):
        self.q.append(path)

    def process_next(self, func):
        if not self.q:
            return None
        path = self.q.popleft()
        return func(path)

    def size(self):
        return len(self.q)
