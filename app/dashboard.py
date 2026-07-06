"""Model performance dashboard for metrics."""

import pandas as pd
import streamlit as st


def render_dashboard(metrics: dict):
    st.title("Performance Dashboard")
    st.metric("Accuracy", f"{metrics.get('accuracy',0):.2%}")
    st.metric("AUC", f"{metrics.get('auc',0):.3f}")
    df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
    st.bar_chart(df.set_index("metric"))
