"""Model registry for version control."""

import json
from datetime import datetime
from pathlib import Path

REGISTRY_FILE = Path("models/registry.json")


def register_model(name: str, path: str, metrics: dict | None = None):
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    registry = []
    if REGISTRY_FILE.exists():
        registry = json.loads(REGISTRY_FILE.read_text())
    entry = {
        "name": name,
        "path": path,
        "metrics": metrics or {},
        "date": datetime.now().isoformat(),
    }
    registry.append(entry)
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2))
    return entry


def list_models():
    if not REGISTRY_FILE.exists():
        return []
    return json.loads(REGISTRY_FILE.read_text())
