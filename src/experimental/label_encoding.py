"""Label encoding for disease categories."""
from sklearn.preprocessing import LabelEncoder
CLASSES = ["normal","pneumonia","covid","tumor"]
_encoder = LabelEncoder()
_encoder.fit(CLASSES)
def encode_label(label: str) -> int:
    return int(_encoder.transform([label])[0])
def decode_label(idx: int) -> str:
    return str(_encoder.inverse_transform([idx])[0])
