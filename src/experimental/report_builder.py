"""Streamlining report generation with templates."""

from pathlib import Path

from src.report import generate_report


def build_report(patient_id, prediction, include_heatmap=False):
    output = f"reports/report_{patient_id}.pdf"
    Path("reports").mkdir(exist_ok=True)
    heatmap_path = prediction.get("heatmap_path") if include_heatmap else None
    return generate_report(patient_id, prediction, output, heatmap_path)
