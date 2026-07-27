"""Cached data loaders and pure helpers for the Phase K Streamlit app.

No Streamlit widget calls live here - only `st.cache_data`/`st.cache_resource`
decorated loaders and plain functions, so this module stays importable and
unit-testable without a Streamlit runtime. Widget code lives in app/pages/.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

DATA_PROCESSED = Path("data/processed")


@st.cache_data
def load_inference_contract() -> dict:
    """Load the C-phase deployment contract: model path, threshold, required_columns schema."""
    with open(DATA_PROCESSED / "c_phase_inference_contract.json") as f:
        return json.load(f)


@st.cache_data
def load_model_card() -> dict:
    """Load the binary champion's model card (val/test metrics, confusion matrices)."""
    with open(DATA_PROCESSED / "a3_binary_model_card.json") as f:
        return json.load(f)


@st.cache_data
def load_binary_comparison() -> pd.DataFrame:
    """Load the 10-candidate binary-KSI comparison table."""
    return pd.read_csv(DATA_PROCESSED / "a3_binary_model_comparison.csv")


@st.cache_data
def load_3class_comparison() -> pd.DataFrame:
    """Load the 19-configuration 3-class comparison table (the pre-reframe ceiling evidence)."""
    return pd.read_csv(DATA_PROCESSED / "a3_model_comparison.csv")


@st.cache_data
def load_candidate_metrics() -> pd.DataFrame:
    """Load the C-phase candidate metrics table with confusion matrices parsed to lists."""
    df = pd.read_csv(DATA_PROCESSED / "c_phase_candidate_metrics.csv")
    df["confusion_matrix"] = df["confusion_matrix"].apply(ast.literal_eval)
    return df


@st.cache_data
def load_permutation_importance(
    model_name: str = "binary_random_forest_balanced", top_n: int = 15
) -> pd.DataFrame:
    """Load global permutation importance for one model, sorted by rank ascending."""
    df = pd.read_csv(DATA_PROCESSED / "c_phase_permutation_importance.csv")
    df = df[df["model"] == model_name].sort_values("rank").head(top_n)
    return df.reset_index(drop=True)
