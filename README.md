# Medical Computer Vision - Radiology Assistant

An intelligent assistant for analyzing radiology images (X-Ray / MRI) using deep learning.

## Overview
This system receives medical images (e.g., lung X-Ray for pneumonia/COVID detection or brain MRI for tumor detection) and highlights suspicious regions using deep learning models.

## Features
- Classification of medical images (healthy vs diseased) using pre-trained CNNs (ResNet50 / EfficientNet)
- Explainable AI with Grad-CAM heatmaps to show model focus areas
- Interactive GUI for image upload and visualization
- Confidence scores and PDF diagnostic report generation

## Tech Stack
- Python, PyTorch, TensorFlow, OpenCV
- Streamlit / PyQt6 for GUI

## Usage
```bash
pip install -r requirements.txt
streamlit run app/main.py
# training
python -m src.train --data data --epochs 10
# batch predict
python -m src.batch_predict data/test
```

## Project Structure
- `src/preprocessing.py` - DICOM/PIL loading and normalization
- `src/model.py` - ResNet50 classifier
- `src/gradcam.py` - explainable heatmaps
- `src/inference.py` - prediction with confidence
- `app/main.py` - file uploader widget for patient scans + Grad-CAM + PDF
- `app/sidebar.py` - model selection, threshold, heatmap toggle

## How to Finish / Verify
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/test_syntax.py -v   # syntax check without GPU
pytest tests/test_preprocessing.py -v  # needs torch+cv2
streamlit run app/main.py        # http://localhost:8501
uvicorn src.api:app --reload     # http://localhost:8000/predict
```
Place data as `data/<class>/*.jpg` or `.dcm` (see `data/README.md`). Train: `python -m src.train --data data` saves `models/resnet50_medical.pth`. Evaluate: `python -m src.evaluate`.

## Status
MVP wired: upload -> `predict_with_heatmap()` -> confidence + Grad-CAM overlay (`src/visualization.py`) -> PDF (`src/report.py`) -> sidebar settings. 130+ modules scaffolded for training/eval, security, deployment (Docker, CI).
