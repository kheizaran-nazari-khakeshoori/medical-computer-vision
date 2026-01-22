"""DICOM viewer for streamlit app."""
import streamlit as st
from src.preprocessing import load_image
import tempfile
def render_dicom_viewer(uploaded_file):
    if uploaded_file.name.endswith((".dcm",".dicom")):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
            tmp.write(uploaded_file.getvalue())
            img = load_image(tmp.name)
            st.image(img, caption="DICOM Preview", use_container_width=True)
    else:
        st.warning("not a dicom file")
