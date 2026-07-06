"""Managing patient queue with priority."""

import heapq
from datetime import datetime


class PatientQueue:
    def __init__(self):
        self.q = []

    def push(self, patient_id, priority=1):
        heapq.heappush(self.q, (priority, datetime.now(), patient_id))

    def pop(self):
        return heapq.heappop(self.q)[2] if self.q else None

    def size(self):
        return len(self.q)
