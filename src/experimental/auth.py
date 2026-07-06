"""User authentication for patient data."""

import hashlib
import secrets

_users = {}


def register_user(username: str, password: str):
    salt = secrets.token_hex(8)
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    _users[username] = {"salt": salt, "hash": h}
    return True


def verify_user(username: str, password: str) -> bool:
    if username not in _users:
        return False
    salt = _users[username]["salt"]
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    return h == _users[username]["hash"]
