"""Patient anonymization for privacy protection."""

import pydicom
from pathlib import Path
import hashlib


def anonymize_dicom(input_path: str, output_path: str) -> str:
    ds = pydicom.dcmread(input_path)
    # remove PHI
    for tag in ["PatientName", "PatientID", "PatientBirthDate"]:
        if tag in ds:
            ds[tag].value = "ANONYMIZED"
    # hash ID
    if hasattr(ds, "PatientID"):
        ds.PatientID = hashlib.sha256(str(ds.PatientID).encode()).hexdigest()[:8]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(output_path)
    return output_path


def anonymize_patient_id(patient_id: str) -> str:
    return hashlib.sha256(patient_id.encode()).hexdigest()[:8]
