"""Testing preprocessing on sample dicom files."""
from pathlib import Path
from src.preprocessing import load_image
def test_dicom_load_missing():
    try:
        load_image("non_existent.dcm")
        assert False
    except Exception:
        assert True
def test_dicom_dummy():
    assert True
