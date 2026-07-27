"""Model Comparison page: candidate table, Pareto front, confusion matrix, robustness."""

import pandas as pd
import streamlit as st

from unfallatlas.viz.metrics_viz import plot_binary_f1_recall_front, plot_confusion_matrix_heatmap
from unfallatlas.viz.streamlit_app import (
    load_binary_comparison,
    load_candidate_metrics,
    load_inference_contract,
    load_model_card,
)

st.title("Model Comparison")

st.subheader("All 10 candidates (binary KSI)")
candidate_df = load_candidate_metrics()
st.dataframe(
    candidate_df[
        [
            "model",
            "family",
            "evaluation_role",
            "macro_f1",
            "recall_ksi",
            "recall_slight",
            "latency_ms_per_1k",
        ]
    ],
    use_container_width=True,
)

st.subheader("Pareto front: macro-F1 vs. Recall(KSI)")
st.plotly_chart(plot_binary_f1_recall_front(load_binary_comparison()), use_container_width=True)

st.subheader("Champion confusion matrix (test 2024)")
card = load_model_card()
confusion_matrix = card["test_2024_metrics"]["confusion_matrix"]
st.plotly_chart(
    plot_confusion_matrix_heatmap(confusion_matrix, labels=["KSI", "slight"]),
    use_container_width=True,
)

st.subheader("Finalist comparison: macro-F1, latency, robustness")
contract = load_inference_contract()
finalists_df = pd.DataFrame(contract["decision_evidence"]["finalist_measurements"])
st.dataframe(
    finalists_df[
        [
            "model",
            "macro_f1",
            "recall_ksi",
            "latency_ms_per_1k",
            "robustness_score",
            "robustness_status",
        ]
    ],
    use_container_width=True,
)

st.info(contract["decision_evidence"]["preference_conclusion"]["statement"])

with st.expander("Warum bleibt Random Forest der Champion? (German explanation)"):
    st.markdown(
        "**Deutsch:** Auf den hier gemessenen Validierungsdaten (2023) schneidet "
        "XGBoost über die kombinierten Kriterien (Recall, Latenz, Robustheit) besser "
        "ab als Random Forest — Random Forest wäre also nicht der bevorzugte Finalist, "
        "wenn man ausschließlich diese Kennzahlen vergleicht. Random Forest bleibt "
        "trotzdem das ausgelieferte Champion-Modell: Die Modellauswahl wurde bereits "
        "vorher auf Basis der Validierungsdaten getroffen, und der separate "
        "Testdatensatz (2024) darf danach nicht mehr benutzt werden, um zwischen "
        "Kandidaten zu wählen - sonst waeren die fuer Random Forest bereits "
        "berichteten Testmetriken nicht mehr unabhaengig (Data Leakage durch "
        "Overfitting auf den Testdatensatz). Die urspruengliche Entscheidung fuer "
        "Random Forest bleibt deshalb bestehen, auch wenn dieser spaetere "
        "Robustheits-/Latenzvergleich XGBoost knapp vorne sieht.\n\n"
        "**English gloss:** On the measured validation data, XGBoost beats Random "
        "Forest on the combined criteria (recall, latency, robustness) - so Random "
        "Forest isn't the preferred finalist by these numbers alone. It remains the "
        "deployment champion anyway because model selection was already locked in "
        "earlier using validation data; the held-out 2024 test set can't be reused "
        "afterward to re-pick between candidates without invalidating the "
        "independence of the test metrics already reported for Random Forest "
        "(that would be data leakage from overfitting the model-selection process "
        "to the test set). So the original Random Forest choice stands, even though "
        "this later robustness/latency comparison shows XGBoost slightly ahead."
    )
