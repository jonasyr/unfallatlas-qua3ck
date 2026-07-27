"""Overview page: champion headline metrics and the 3-class vs. binary ceiling story."""

import streamlit as st
from streamlit_folium import st_folium

from unfallatlas.viz.metrics_viz import plot_binary_f1_recall_front, plot_f1_recall_front
from unfallatlas.viz.streamlit_app import (
    LIMITATIONS_TEXT,
    build_severity_map,
    load_3class_comparison,
    load_binary_comparison,
    load_model_card,
)

st.title("Unfallatlas KSI Risk Console")
st.caption("Binary KSI (killed or seriously injured) vs. slight-injury severity classification")

card = load_model_card()
test_metrics = card["test_2024_metrics"]

col1, col2, col3 = st.columns(3)
col1.metric("Macro-F1 (test 2024)", f"{test_metrics['macro_f1']:.3f}")
col2.metric("Recall (KSI)", f"{test_metrics['recall_ksi']:.3f}")
col3.metric("Decision threshold", f"{card['optimal_threshold_val_2023']:.4f}")

st.markdown("---")
st.subheader("Why the target was reframed: the 3-class ceiling vs. the binary reframe")
st.markdown(
    "The original 3-class target (killed / seriously injured / slightly injured) has an "
    "empirical ceiling of macro-F1 = 0.424 across 19 configurations, well below the 0.55 "
    "acceptance gate. Reframing as binary KSI (killed-or-seriously-injured vs. slight) "
    "clears both acceptance gates on the held-out 2024 test set."
)

col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(plot_f1_recall_front(load_3class_comparison()), width="stretch")
with col_b:
    st.plotly_chart(plot_binary_f1_recall_front(load_binary_comparison()), width="stretch")

st.markdown("---")
st.subheader("Where accidents happen: severity by location")
st.caption(
    "Each marker aggregates accidents within a ~0.1 degree (~11 km) grid cell. "
    "Red = KSI (killed/seriously injured) accidents are the majority in that "
    "cell - this marks the local *share*, not the accident count, so an "
    "isolated red cell can be a low-traffic rural spot rather than a hotspot. "
    "Teal = slight-injury accidents are the majority. Circle size scales with "
    "the cell's total accident count. Use the layer control (top-right on the "
    "map) to show or hide each severity class."
)
severity_map = build_severity_map()
st_folium(severity_map, height=720, width=None, key="overview_severity_map", returned_objects=[])

with st.expander("Limitations"):
    st.markdown(LIMITATIONS_TEXT)

st.markdown(
    "[View the full Q/U/A3/C phase notebooks (GitHub Pages)]"
    "(https://jonasyr.github.io/unfallatlas-qua3ck/)"
)
