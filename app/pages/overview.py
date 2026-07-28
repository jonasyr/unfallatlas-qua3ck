"""Overview page: champion headline metrics and the 3-class vs. binary ceiling story."""

import folium
import streamlit as st
from streamlit_folium import st_folium

from unfallatlas.viz.metrics_viz import plot_binary_f1_recall_front, plot_f1_recall_front
from unfallatlas.viz.streamlit_app import (
    LIMITATIONS_TEXT,
    build_severity_base_map,
    build_severity_feature_groups,
    load_3class_comparison,
    load_binary_comparison,
    load_model_card,
    severity_legend_markdown,
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
st.subheader("Where severe accidents concentrate")
st.caption(
    "Each circle aggregates the accidents inside a ~0.1 degree (~11 km) cell, drawn "
    "at the mean position of those accidents. Color shows how the cell's share of "
    "KSI (killed/seriously injured) accidents compares against the national average, "
    "not how many accidents it has - use the layer control to isolate a single risk "
    "band. Note the inversion this reveals: the lowest-risk cells carry a median of "
    "1,339 accidents each while the highest-risk cells carry 91. Dense urban areas "
    "produce many mostly-slight collisions; rural roads produce far fewer that are "
    "far more often severe."
)
st_folium(
    build_severity_base_map(),
    feature_group_to_add=build_severity_feature_groups(),
    # streamlit-folium 0.27's `layer_control` parameter takes a folium.LayerControl
    # instance (not a bool) and attaches it to the map itself, internally, with its
    # own variable-rewriting - we never call .add_to() on it ourselves.
    layer_control=folium.LayerControl(),
    height=720,
    width=None,
    key="overview_severity_map",
    returned_objects=[],
)
st.markdown(severity_legend_markdown(), unsafe_allow_html=True)

with st.expander("Limitations"):
    st.markdown(LIMITATIONS_TEXT)

st.markdown(
    "[View the full Q/U/A3/C phase notebooks (GitHub Pages)]"
    "(https://jonasyr.github.io/unfallatlas-qua3ck/)"
)
