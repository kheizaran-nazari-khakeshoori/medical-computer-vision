"""Storing predictions in database."""
import json
from pathlib import Path
DB = Path("data/predictions.json")
def store_prediction(patient_id, result: dict):
    DB.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(DB.read_text()) if DB.exists() else []
    data.append({"patient_id": patient_id, **result})
    DB.write_text(json.dumps(data, indent=2))
    return True
