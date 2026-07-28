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
    load_national_ksi_rate,
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
st.subheader("Where accidents are disproportionately severe")
st.caption(
    "Each circle aggregates the accidents inside a ~0.1 degree (~11 km) cell, drawn "
    "at the mean position of those accidents. Color shows how the cell's share of "
    "KSI (killed/seriously injured) accidents compares against the national average, "
    "not how many accidents it has - use the layer control to isolate a single risk "
    "band. Note the inversion this reveals: the lowest-risk cells carry a median of "
    # Measured constants from the committed dataset/grid - update if the dataset
    # or the 0.1-degree grid precision changes.
    "1,339 accidents each while the highest-risk cells carry 91. The most likely "
    "explanation: dense urban areas produce many mostly-slight collisions, while "
    "rural roads produce far fewer accidents that are far more often severe."
)
st_folium(
    build_severity_base_map(),
    feature_group_to_add=build_severity_feature_groups(),
    # streamlit-folium 0.27's `layer_control` parameter takes a folium.LayerControl
    # instance (not a bool) and attaches it to the map itself, internally, with its
    # own variable-rewriting - we never call .add_to() on it ourselves. This
    # internal attachment is exactly why build_severity_base_map() must not be
    # cached: caching the map would let this attachment (and the FeatureGroups'
    # own internal .add_to() calls) persist into the next render, baking in
    # stale layer identifiers and reproducing the ReferenceError blank-map bug.
    # build_severity_feature_groups() is likewise never cached (nothing handed to
    # st_folium ever is) - see its docstring for the cross-session variant of the
    # same failure.
    layer_control=folium.LayerControl(),
    height=720,
    width=None,
    key="overview_severity_map",
    returned_objects=[],
)
st.markdown(severity_legend_markdown(load_national_ksi_rate()), unsafe_allow_html=True)

with st.expander("Limitations"):
    st.markdown(LIMITATIONS_TEXT)
    st.markdown(
        "**Map-specific limitations**\n\n"
        "- Shrinkage is a deliberate bias, not an accident: it deliberately pulls "
        "thinly-sampled cells toward the national average, so a genuinely "
        "dangerous cell with only a few recorded accidents will read as less "
        "severe than it may actually be. The opacity channel exists to keep that "
        "uncertainty visible instead of hiding it.\n"
        "- The denominator is reporting-driven, not just traffic-driven: the "
        "dataset holds police-reported personal-injury accidents only, so a "
        "cell's total reflects both how much traffic passes through it and how "
        "consistently accidents there get reported.\n"
        "- Grid cells are not administrative units: a ~11 km cell can straddle a "
        "city boundary and a rural road, blending two very different severity "
        "regimes into a single number.\n"
        "- Relative risk is not causal: a cell sitting at 2x the national rate is "
        "not thereby shown to be dangerous *because of* its road layout - only "
        "that its recorded outcomes skew more severe than average."
    )

st.markdown(
    "[View the full Q/U/A3/C phase notebooks (GitHub Pages)]"
    "(https://jonasyr.github.io/unfallatlas-qua3ck/)"
)
