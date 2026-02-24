"""Enabling auto-reload for development."""
import os
def enable_reload():
    os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "true"
    return True
