"""Streamlit app for radiology assistant."""

import sys
from pathlib import Path

# ensure project root on sys.path for `from src...` when run via `streamlit run app/main.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from app.sidebar import render_sidebar
from src.inference import predict_with_heatmap
from src.preprocessing import load_image, preprocess_image
from src.visualization import overlay_heatmap


@st.cache_resource(show_spinner=False)
def get_cached_model():
    from pathlib import Path

    from src.config import MODEL_PATH
    from src.model import get_model

    ckpt = Path(MODEL_PATH)
    if ckpt.exists():
        try:
            from src.model import load_model

            return load_model(str(ckpt), device="cpu")
        except Exception:
            pass
    return get_model(pretrained=True)


st.set_page_config(page_title="Radiology Assistant", layout="wide")
settings = render_sidebar()
st.title("Medical Computer Vision - Radiology Assistant")
st.write("Upload a chest X-Ray or brain MRI to get classification and Grad-CAM visualization.")
st.caption(
    f"Active model: {settings['model']} | threshold: {settings['threshold']} | heatmap: {settings['show_heatmap']}"
)

uploaded_file = st.file_uploader(
    "added file uploader widget for patient scans", type=["jpg", "jpeg", "png", "dcm", "dicom"]
)

col1, col2 = st.columns(2)

if uploaded_file is not None:
    try:
        # handle DICOM vs standard image
        tmp_paths = []
        if uploaded_file.name.lower().endswith((".dcm", ".dicom")):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            tmp_paths.append(tmp_path)
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
                        cached_model = get_cached_model()
                        result = predict_with_heatmap(image, model=cached_model)
                        label = result["label"]
                        conf = result["confidence"]
                        st.metric("Prediction", label, f"{conf:.2%} confidence")
                        st.write("Probabilities:")
                        for cls, p in result["probabilities"].items():
                            st.progress(float(p), text=f"{cls}: {p:.2%}")
                        if conf < settings["threshold"]:
                            st.warning("Low confidence — recommend radiologist review")
                        heatmap = result.get("heatmap")
                        if heatmap is not None and settings["show_heatmap"]:
                            overlay = overlay_heatmap(image, heatmap)
                            st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
                            st.session_state["last_result"] = result
                            st.session_state["last_image"] = image
                        else:
                            st.info("Heatmap not available")
                        # PDF report download
                        try:
                            from src.report import generate_report

                            patient_id = uploaded_file.name.split(".")[0][:20] or "patient_001"
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".pdf"
                            ) as tmp_pdf:
                                pdf_path = tmp_pdf.name
                            tmp_paths.append(pdf_path)
                            # save heatmap temporarily for report
                            heatmap_path = None
                            if heatmap is not None:
                                heat_color = cv2.applyColorMap(
                                    np.uint8(255 * heatmap), cv2.COLORMAP_JET
                                )
                                heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
                                with tempfile.NamedTemporaryFile(
                                    delete=False, suffix=".png"
                                ) as tmp_img:
                                    Image.fromarray(heat_color).save(tmp_img.name)
                                    heatmap_path = tmp_img.name
                                tmp_paths.append(heatmap_path)
                            generate_report(patient_id, result, pdf_path, heatmap_path)
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    "Download PDF report",
                                    f,
                                    file_name=f"report_{patient_id}.pdf",
                                    mime="application/pdf",
                                )
                        except Exception as pdf_e:
                            st.caption(f"PDF generation: {pdf_e}")
                        finally:
                            # cleanup temp files for smoother run
                            for p in tmp_paths:
                                try:
                                    Path(p).unlink(missing_ok=True)
                                except Exception:
                                    pass
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
