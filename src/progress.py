"""Progress bar for long running jobs."""
import time
def progress_bar(iterable, total=None, desc="processing"):
    total = total or len(iterable)
    for i, item in enumerate(iterable):
        print(f"{desc}: {i+1}/{total}")
        yield item
