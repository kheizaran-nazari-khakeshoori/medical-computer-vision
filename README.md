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
