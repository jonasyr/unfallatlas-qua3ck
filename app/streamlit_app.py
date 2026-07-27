"""Phase K Streamlit entry point: page config and navigation wiring only.

All data loading, model inference, and plotting logic lives in
src/unfallatlas/viz/streamlit_app.py and the individual page modules under
app/pages/ - this file stays a thin wiring layer.
"""

import sys
from pathlib import Path

# Prepend src/ so pages always import the repo's current `unfallatlas` source,
# never a stale cached install. Streamlit Community Cloud reuses its pip
# environment across deploys and can skip reinstalling this project's own
# package (pyproject.toml's version doesn't change on every push), which has
# caused "cannot import name X" errors for symbols that are genuinely present
# in the committed source - this makes that class of staleness impossible
# regardless of what's sitting in site-packages.
_SRC_DIR = str(Path(__file__).parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import streamlit as st  # noqa: E402

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
