"""Image comparison slider for before after."""
import streamlit as st
from PIL import Image
def image_comparison(original: Image.Image, processed: Image.Image):
    st.image(original, caption="Original", use_container_width=True)
    st.image(processed, caption="Processed", use_container_width=True)
    st.caption("side-by-side comparison")
