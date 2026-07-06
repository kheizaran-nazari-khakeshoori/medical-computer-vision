"""Sidebar controls for model settings."""

import streamlit as st

from src.config import CLASS_NAMES, MODEL_NAME


def render_sidebar():
    st.sidebar.title("Settings")
    model = st.sidebar.selectbox(
        "Model", ["resnet50", "efficientnet_b0"], index=0 if MODEL_NAME == "resnet50" else 1
    )
    threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    show_heatmap = st.sidebar.checkbox("Show Grad-CAM heatmap", value=True)
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Classes: {', '.join(CLASS_NAMES)}")
    return {"model": model, "threshold": threshold, "show_heatmap": show_heatmap}
