"""Tracking model performance over time."""

import json
from datetime import datetime
from pathlib import Path

TRACK = Path("logs/performance.json")


def track(metrics: dict):
    TRACK.parent.mkdir(parents=True, exist_ok=True)
    logs = json.loads(TRACK.read_text()) if TRACK.exists() else []
    logs.append({"time": datetime.now().isoformat(), **metrics})
    TRACK.write_text(json.dumps(logs, indent=2))
