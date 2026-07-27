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

import duckdb
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

DATA_PROCESSED = Path("data/processed")
ACCIDENTS_PARQUET = Path("data/accidents.parquet")

DEFAULT_DWD_STATION_ID = "01975"
DEFAULT_H3_CELL = "881f15ad31fffff"
DEFAULT_DWD_STATION_DIST_KM = 9.51

SEVERITY_COLORS = {"KSI": "#E63946", "slight": "#2A9D8F"}

LIMITATIONS_TEXT = (
    "- The strongest available feature (accident type, `UART`) has a Cramer's V "
    "of only 0.18 against the target - even the best predictors have weak "
    "individual association with severity.\n"
    "- No demographic data (age, seatbelt use) or impact-speed data is in the "
    "Unfallatlas dataset, even though the literature identifies these as the "
    "strongest severity predictors.\n"
    "- Correlation is not causation: the model finds statistical association, "
    "not a causal mechanism.\n"
    "- The 'Why This Prediction' page shows global permutation importance, "
    "not a per-instance explanation - no SHAP was computed for this project."
)

WEEKDAY_LABELS = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday",
}
LICHTVERH_LABELS = {0: "Daylight", 1: "Dusk/Dawn", 2: "Darkness"}
STRZUSTAND_LABELS = {0: "Dry", 1: "Wet/Slippery", 2: "Wintry"}

DEFAULT_WIDGET_VALUES = {
    "UREGBEZ": "1",
    "UKREIS": "01",
    "UMONAT": 6,
    "USTUNDE": 12,
    "UWOCHENTAG": 2,
    "UART": 0,
    "UTYP1": 1,
    "ULICHTVERH": 0,
    "STRZUSTAND": 0,
    "IstRad": False,
    "IstPKW": True,
    "IstFuss": False,
    "IstKrad": False,
    "IstGkfz": False,
    "IstSonstig": False,
    "LON": 9.67,
    "LAT": 50.85,
    "dwd_station_id": DEFAULT_DWD_STATION_ID,
    "dwd_station_dist_km": DEFAULT_DWD_STATION_DIST_KM,
    "dwd_temp_air_2m": 10.0,
    "dwd_precip_mm": 0.0,
    "dwd_visibility_m": 10000.0,
    "dwd_wind_speed_ms": 3.0,
    "_precip_bucket": "dry (0 mm)",
    "h3_cell": DEFAULT_H3_CELL,
    "osm_dominant_road_class": "residential",
    "osm_maxspeed_mean": 50.0,
    "osm_maxspeed_max": 50.0,
    "osm_road_density": 100.0,
    "osm_way_count": 50.0,
}


@st.cache_data
def load_inference_contract() -> dict:
    """Load the C-phase deployment contract: model path, threshold, required_columns schema."""
    try:
        with open(DATA_PROCESSED / "c_phase_inference_contract.json") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load c_phase_inference_contract.json")
        raise


@st.cache_data
def load_categorical_options(column: str) -> list[str]:
    """Load sorted distinct values for a high-cardinality column with no fixed
    category list in the inference contract (currently only 'UKREIS').

    Reads data/accidents.parquet directly via DuckDB (a single-column
    columnar scan, not a full-dataset load); this file is committed and
    Git-LFS-tracked, so no notebook execution is required.
    """
    try:
        con = duckdb.connect()
        query = f"SELECT DISTINCT {column} FROM '{ACCIDENTS_PARQUET}' ORDER BY {column}"  # noqa: S608
        return con.execute(query).df()[column].astype(str).tolist()
    except Exception:
        logger.exception(f"Failed to load categorical options for column {column!r}")
        raise


def get_column_spec(contract: dict, name: str) -> dict:
    """Return the required_columns entry for one column name."""
    for col in contract["required_columns"]:
        if col["name"] == name:
            return col
    raise KeyError(f"Column {name!r} not found in inference contract required_columns")


@st.cache_data
def load_model_card() -> dict:
    """Load the binary champion's model card (val/test metrics, confusion matrices)."""
    try:
        with open(DATA_PROCESSED / "a3_binary_model_card.json") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load a3_binary_model_card.json")
        raise


@st.cache_data
def load_binary_comparison() -> pd.DataFrame:
    """Load the 10-candidate binary-KSI comparison table."""
    try:
        return pd.read_csv(DATA_PROCESSED / "a3_binary_model_comparison.csv")
    except Exception:
        logger.exception("Failed to load a3_binary_model_comparison.csv")
        raise


@st.cache_data
def load_3class_comparison() -> pd.DataFrame:
    """Load the 19-configuration 3-class comparison table (the pre-reframe ceiling evidence)."""
    try:
        return pd.read_csv(DATA_PROCESSED / "a3_model_comparison.csv")
    except Exception:
        logger.exception("Failed to load a3_model_comparison.csv")
        raise


@st.cache_data
def load_candidate_metrics() -> pd.DataFrame:
    """Load the C-phase candidate metrics table with confusion matrices parsed to lists."""
    try:
        df = pd.read_csv(DATA_PROCESSED / "c_phase_candidate_metrics.csv")
        df["confusion_matrix"] = df["confusion_matrix"].apply(ast.literal_eval)
        return df
    except Exception:
        logger.exception("Failed to load c_phase_candidate_metrics.csv")
        raise


@st.cache_data
def load_permutation_importance(
    model_name: str = "binary_random_forest_balanced", top_n: int = 15
) -> pd.DataFrame:
    """Load global permutation importance for one model, sorted by rank ascending."""
    try:
        df = pd.read_csv(DATA_PROCESSED / "c_phase_permutation_importance.csv")
        df = df[df["model"] == model_name].sort_values("rank").head(top_n)
        return df.reset_index(drop=True)
    except Exception:
        logger.exception("Failed to load c_phase_permutation_importance.csv")
        raise
