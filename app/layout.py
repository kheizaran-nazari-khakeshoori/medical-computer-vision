"""Restructuring app layout for better ux."""
import streamlit as st
def render_header():
    st.markdown("# Radiology Assistant")
    st.markdown("---")
def render_footer():
    st.markdown("---")
    st.caption("AI assist - not a medical diagnosis")
