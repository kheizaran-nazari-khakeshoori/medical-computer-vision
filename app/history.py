"""History page for patient scan tracking."""

import json
from pathlib import Path

import streamlit as st

HISTORY_FILE = Path("data/history.json")


def save_history(entry: dict):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if HISTORY_FILE.exists():
        history = json.loads(HISTORY_FILE.read_text())
    history.append(entry)
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


def load_history():
    if not HISTORY_FILE.exists():
        return []
    return json.loads(HISTORY_FILE.read_text())


def render_history_page():
    st.title("Patient Scan History")
    history = load_history()
    if not history:
        st.info("no scans yet")
        return
    for entry in reversed(history[-20:]):
        st.write(entry)
