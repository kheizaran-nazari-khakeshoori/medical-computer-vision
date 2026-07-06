"""Audit log for compliance tracking."""

import json
from datetime import datetime
from pathlib import Path

AUDIT_FILE = Path("logs/audit.json")


def log_action(user: str, action: str, resource: str):
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "user": user,
        "action": action,
        "resource": resource,
        "time": datetime.now().isoformat(),
    }
    logs = []
    if AUDIT_FILE.exists():
        logs = json.loads(AUDIT_FILE.read_text())
    logs.append(entry)
    AUDIT_FILE.write_text(json.dumps(logs, indent=2))
    return entry
