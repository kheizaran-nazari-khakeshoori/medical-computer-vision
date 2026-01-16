"""DICOM metadata extraction for reports."""

import pydicom
from pathlib import Path


def extract_metadata(dicom_path: str) -> dict:
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    fields = ["PatientID", "PatientAge", "PatientSex", "Modality", "StudyDate", "BodyPartExamined", "Manufacturer"]
    meta = {}
    for f in fields:
        meta[f] = str(getattr(ds, f, "unknown"))
    return meta


def format_metadata_for_report(meta: dict) -> str:
    return "\n".join([f"{k}: {v}" for k, v in meta.items()])
