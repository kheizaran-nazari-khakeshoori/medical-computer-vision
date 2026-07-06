"""Recording inference latency."""

import json
from datetime import datetime
from pathlib import Path

LOG = Path("logs/latency.json")


def record_latency(latency_ms: float, model="resnet50"):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    logs = json.loads(LOG.read_text()) if LOG.exists() else []
    logs.append({"latency_ms": latency_ms, "model": model, "time": datetime.now().isoformat()})
    LOG.write_text(json.dumps(logs, indent=2))
