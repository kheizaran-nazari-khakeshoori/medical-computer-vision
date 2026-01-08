"""Streamlit app for radiology assistant."""

import io
import streamlit as st
from PIL import Image

from src.config import CLASS_NAMES
from src.preprocessing import preprocess_image


st.set_page_config(page_title="Radiology Assistant", layout="wide")
st.title("Medical Computer Vision - Radiology Assistant")
st.write("Upload a chest X-Ray or brain MRI to get classification and Grad-CAM visualization.")

uploaded_file = st.file_uploader("added file uploader widget for patient scans", type=["jpg", "jpeg", "png", "dcm", "dicom"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    try:
        # handle DICOM vs standard image
        if uploaded_file.name.lower().endswith((".dcm", ".dicom")):
            import tempfile
            from src.preprocessing import load_image

            with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            image = load_image(tmp_path)
        else:
            image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")

        with col1:
            st.subheader("Patient Scan")
            st.image(image, use_container_width=True)
            st.caption(f"File: {uploaded_file.name} | Size: {image.size}")

        with col2:
            st.subheader("Analysis")
            st.info("Model inference will appear here once trained weights are loaded.")
            # placeholder for inference
            if st.button("Run diagnosis"):
                tensor = preprocess_image(image)
                st.success(f"Preprocessed tensor shape: {tuple(tensor.shape)}")
                st.write(f"Classes: {CLASS_NAMES}")
                st.warning("Load trained model in src/model.py to enable real predictions + Grad-CAM overlay.")

    except Exception as e:
        st.error(f"Failed to load image: {e}")
else:
    st.info("Please upload a patient scan to begin.")
    st.caption("Supported formats: JPG, PNG, DICOM (.dcm)")
