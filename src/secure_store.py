"""Securing patient data with encryption."""
from src.encryption import encrypt_file, generate_key
from pathlib import Path
def secure_save(data: bytes, path: str, key: bytes = None):
    key = key or generate_key()
    token = encrypt_file(data, key)
    Path(path).write_bytes(token)
    return key
