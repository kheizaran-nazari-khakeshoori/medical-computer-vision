"""Recording training history with csv logger."""
import csv
from pathlib import Path
def log_epoch(csv_path: str, epoch: int, loss: float, acc: float):
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(["epoch","loss","accuracy"])
        w.writerow([epoch, loss, acc])
