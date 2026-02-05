"""Monitoring model drift with performance metrics."""
import json
from pathlib import Path
from datetime import datetime
LOG = Path("logs/drift.json")
def log_performance(accuracy: float, dataset: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"accuracy": accuracy, "dataset": dataset, "time": datetime.now().isoformat()}
    logs = json.loads(LOG.read_text()) if LOG.exists() else []
    logs.append(entry)
    LOG.write_text(json.dumps(logs, indent=2))
    return entry
def check_drift(threshold=0.05):
    if not LOG.exists(): return False
    logs = json.loads(LOG.read_text())
    if len(logs) < 2: return False
    return abs(logs[-1]["accuracy"] - logs[-2]["accuracy"]) > threshold
