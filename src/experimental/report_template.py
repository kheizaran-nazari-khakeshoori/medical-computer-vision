"""Patient report template for pdf export."""

from src.report import generate_report


def create_patient_report(patient_id, prediction, output="report.pdf"):
    return generate_report(patient_id, prediction, output)


TEMPLATE_FIELDS = ["patient_id", "prediction", "confidence", "heatmap", "date"]
