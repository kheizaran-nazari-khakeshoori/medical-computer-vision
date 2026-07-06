"""Creating data validation report."""

import json
from pathlib import Path


def create_validation_report(dataset_path: str, output="reports/validation.json"):
    p = Path(dataset_path)
    files = list(p.rglob("*.jpg")) + list(p.rglob("*.png")) + list(p.rglob("*.dcm"))
    report = {"total_files": len(files), "by_ext": {}}
    for f in files:
        report["by_ext"][f.suffix] = report["by_ext"].get(f.suffix, 0) + 1
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2))
    return report
