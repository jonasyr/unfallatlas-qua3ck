"""Phase K Streamlit entry point: page config and navigation wiring only.

All data loading, model inference, and plotting logic lives in
src/unfallatlas/viz/streamlit_app.py and the individual page modules under
app/pages/ - this file stays a thin wiring layer.
"""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Unfallatlas KSI Risk Console",
    layout="wide",
    page_icon="\U0001f6a6",
)

PAGES_DIR = Path(__file__).parent / "pages"

pages = [
    st.Page(str(PAGES_DIR / "overview.py"), title="Overview", icon=":material/bar_chart:"),
    st.Page(
        str(PAGES_DIR / "risk_predictor.py"), title="Risk Predictor", icon=":material/query_stats:"
    ),
    st.Page(
        str(PAGES_DIR / "why_this_prediction.py"),
        title="Why This Prediction",
        icon=":material/search_insights:",
    ),
    st.Page(
        str(PAGES_DIR / "model_comparison.py"), title="Model Comparison", icon=":material/balance:"
    ),
]

nav = st.navigation(pages)
nav.run()
