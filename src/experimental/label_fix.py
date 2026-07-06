"""Correcting label mismatch in dataset loader."""


def correct_label(label: str) -> str:
    mapping = {
        "pneumonia": "diseased",
        "covid": "diseased",
        "tumor": "diseased",
        "normal": "normal",
    }
    return mapping.get(label.lower(), label)
