"""Handling timezone for audit logs."""

from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def to_local(iso_str):
    from datetime import datetime

    dt = datetime.fromisoformat(iso_str)
    return dt.astimezone().isoformat()
