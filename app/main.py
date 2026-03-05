"""Streamlit app for radiology assistant."""

import io
import tempfile
import streamlit as st
from PIL import Image
import numpy as np

from src.config import CLASS_NAMES
from src.inference import predict_with_heatmap
from src.preprocessing import preprocess_image
from src.visualization import overlay_heatmap


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
            if st.button("Run diagnosis"):
                with st.spinner("Running inference..."):
                    try:
                        result = predict_with_heatmap(image)
                        label = result["label"]
                        conf = result["confidence"]
                        st.metric("Prediction", label, f"{conf:.2%} confidence")
                        st.write("Probabilities:")
                        for cls, p in result["probabilities"].items():
                            st.progress(float(p), text=f"{cls}: {p:.2%}")
                        if conf < 0.6:
                            st.warning("Low confidence — recommend radiologist review")
                        heatmap = result.get("heatmap")
                        if heatmap is not None:
                            overlay = overlay_heatmap(image, heatmap)
                            st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
                            st.session_state["last_result"] = result
                            st.session_state["last_image"] = image
                        else:
                            st.info("Heatmap not available")
                    except Exception as e:
                        st.error(f"Inference failed: {e}")
                        # fallback to preprocessing check
                        tensor = preprocess_image(image)
                        st.write(f"Preprocessed shape: {tuple(tensor.shape)}")

    except Exception as e:
        st.error(f"Failed to load image: {e}")
else:
    st.info("Please upload a patient scan to begin.")
    st.caption("Supported formats: JPG, PNG, DICOM (.dcm)")
