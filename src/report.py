"""PDF diagnostic report generation."""

from datetime import datetime
from pathlib import Path

from fpdf import FPDF


class DiagnosisReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Radiology Assistant - Diagnostic Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")


def generate_report(patient_id: str, prediction: dict, output_path: str = "report.pdf", heatmap_path: str | None = None):
    pdf = DiagnosisReport()
    pdf.add_page()
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Patient ID: {patient_id}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Prediction: {prediction.get('label', 'unknown')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Confidence: {prediction.get('confidence', 0):.2%}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Probabilities:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for cls, prob in prediction.get("probabilities", {}).items():
        pdf.cell(0, 6, f"  - {cls}: {prob:.2%}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 6, "Disclaimer: This is an AI-assisted preview and not a medical diagnosis. Consult a radiologist.")
    if heatmap_path and Path(heatmap_path).exists():
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Grad-CAM Heatmap", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.image(heatmap_path, x=30, w=150)
    pdf.output(output_path)
    return output_path
