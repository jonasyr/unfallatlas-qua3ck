"""Why This Prediction page: global permutation importance + user-input context."""

import pandas as pd
import streamlit as st

from unfallatlas.viz.streamlit_app import (
    decode_feature_value,
    display_feature_name,
    load_permutation_importance,
)

st.title("Why This Prediction")

last_prediction = st.session_state.get("last_prediction")
if last_prediction is None:
    st.info("No prediction yet. Go to the Risk Predictor page and submit a prediction first.")
    st.stop()

st.warning(
    "This shows global, model-level permutation importance from the C-phase analysis, "
    "not a per-instance SHAP explanation. No SHAP was computed for this project."
)

importance_df = load_permutation_importance()
importance_df["display_name"] = importance_df["feature"].apply(display_feature_name)
st.bar_chart(importance_df.set_index("display_name")["importance_mean"])

st.subheader("Your inputs for the globally most influential features")
st.caption(
    "These are the values you submitted for the model's top globally-important "
    "features. This is context, not a causal explanation of this specific prediction."
)
top_features = importance_df["feature"].tolist()
rows = [
    {
        "feature": display_feature_name(feature),
        "your value": decode_feature_value(feature, last_prediction["inputs"][feature]),
    }
    for feature in top_features
    if feature in last_prediction["inputs"]
]
st.table(pd.DataFrame(rows))
