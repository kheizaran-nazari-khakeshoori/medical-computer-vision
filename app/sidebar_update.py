"""Updating sidebar with model selection."""
import streamlit as st
from src.config import CLASS_NAMES
def render_model_selector():
    return st.sidebar.selectbox("choose model", ["resnet50","efficientnet_b0","efficientnet_b3"])
def render_threshold_slider():
    return st.sidebar.slider("confidence threshold", 0.0, 1.0, 0.5)
