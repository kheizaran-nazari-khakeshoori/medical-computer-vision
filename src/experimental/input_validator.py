"""Validating user input for prediction requests."""

from PIL import Image


def validate_upload(file):
    if file is None:
        return False, "no file uploaded"
    if file.size > 50 * 1024 * 1024:
        return False, "file too large"
    try:
        Image.open(file).verify()
        return True, "valid"
    except Exception as e:
        return False, str(e)


def validate_patient_id(pid: str):
    return pid.isalnum() and 3 <= len(pid) <= 20
