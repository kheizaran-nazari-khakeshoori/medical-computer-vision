"""Tracking user feedback for predictions."""
import json
from pathlib import Path
from datetime import datetime
FB = Path("data/feedback.json")
def add_feedback(patient_id, correct_label, comment=""):
    FB.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(FB.read_text()) if FB.exists() else []
    data.append({"patient_id": patient_id, "correct_label": correct_label, "comment": comment, "time": datetime.now().isoformat()})
    FB.write_text(json.dumps(data, indent=2))
