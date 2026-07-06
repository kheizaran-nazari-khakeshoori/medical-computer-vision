"""Needing pacs integration for hospital system."""


# stub for PACS DICOM C-STORE / C-MOVE
def fetch_from_pacs(pacs_url: str, patient_id: str):
    print(f"fetching {patient_id} from {pacs_url} - stub")
    return []


def push_to_pacs(pacs_url: str, dicom_path: str):
    print(f"pushing {dicom_path} to {pacs_url} - stub")
    return True
