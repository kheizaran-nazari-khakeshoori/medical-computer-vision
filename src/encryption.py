"""Data encryption for sensitive scans."""
import hashlib
from cryptography.fernet import Fernet
import base64
def generate_key():
    return Fernet.generate_key()
def encrypt_file(data: bytes, key: bytes) -> bytes:
    f = Fernet(key)
    return f.encrypt(data)
def decrypt_file(token: bytes, key: bytes) -> bytes:
    f = Fernet(key)
    return f.decrypt(token)
