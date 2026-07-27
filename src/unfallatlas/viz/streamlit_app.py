"""Cached data loaders and pure helpers for the Phase K Streamlit app.

No Streamlit widget calls live here - only `st.cache_data`/`st.cache_resource`
decorated loaders and plain functions, so this module stays importable and
unit-testable without a Streamlit runtime. Widget code lives in app/pages/.
"""

from __future__ import annotations

import ast
import json
import logging
import math
from pathlib import Path

import duckdb
import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

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
UART_LABELS = {
    0: "Other accident",
    1: "Collision with stationary vehicle",
    2: "Collision with vehicle ahead",
    3: "Collision with vehicle travelling alongside",
    4: "Collision with oncoming vehicle",
    5: "Collision while turning or crossing",
    6: "Collision with pedestrian",
    7: "Impact with road obstacle",
    8: "Departure from road to the right",
    9: "Departure from road to the left",
}
UTYP1_LABELS = {
    1: "Loss-of-control accident",
    2: "Turning accident",
    3: "Crossing or entering accident",
    4: "Pedestrian crossing accident",
    5: "Stationary-traffic accident",
    6: "Longitudinal-traffic accident",
    7: "Other accident",
}

FEATURE_DISPLAY_NAMES = {
    "IstFuss": "Pedestrian Involved",
    "IstGkfz": "Heavy Goods Vehicle Involved",
    "IstKrad": "Motorcycle Involved",
    "IstPKW": "Car Involved",
    "IstRad": "Cyclist Involved",
    "IstSonstig": "Other Vehicle Involved",
    "LAT": "Latitude",
    "LON": "Longitude",
    "STRZUSTAND": "Road Condition",
    "UART": "Accident Type",
    "UKREIS": "Kreis Code",
    "ULICHTVERH": "Light Conditions",
    "UMONAT": "Month",
    "UREGBEZ": "Regierungsbezirk Code",
    "USTUNDE": "Hour of Day",
    "UTYP1": "Accident Category",
    "UWOCHENTAG": "Weekday",
    "_precip_bucket": "Precipitation Bucket",
    "dwd_precip_mm": "Precipitation (mm)",
    "dwd_station_dist_km": "Weather Station Distance (km)",
    "dwd_station_id": "Weather Station ID",
    "dwd_temp_air_2m": "Air Temperature (C)",
    "dwd_visibility_m": "Visibility (m)",
    "dwd_wind_speed_ms": "Wind Speed (m/s)",
    "h3_cell": "H3 Location Cell",
    "osm_dominant_road_class": "Dominant Road Class",
    "osm_maxspeed_max": "Max Speed Limit (km/h)",
    "osm_maxspeed_mean": "Mean Speed Limit (km/h)",
    "osm_road_density": "Road Density",
    "osm_way_count": "Road Way Count",
}


def display_feature_name(name: str) -> str:
    """Return a human-readable label for a contract column name, or the raw name if unmapped."""
    return FEATURE_DISPLAY_NAMES.get(name, name)


COLUMN_LABEL_MAPS = {
    "UART": UART_LABELS,
    "UTYP1": UTYP1_LABELS,
    "STRZUSTAND": STRZUSTAND_LABELS,
    "ULICHTVERH": LICHTVERH_LABELS,
    "UWOCHENTAG": WEEKDAY_LABELS,
}


def decode_feature_value(feature: str, value):
    """Return a human-readable value for a coded column (e.g. UART code -> label), else value unchanged."""
    return COLUMN_LABEL_MAPS.get(feature, {}).get(value, value)


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


@st.cache_data
def nearest_uregbez_ukreis(lat: float, lon: float) -> tuple[str, str]:
    """Find the UREGBEZ/UKREIS of the accident record nearest to a map click.

    This dataset has no ULAND (Bundesland) column anywhere - not just excluded
    from the model's feature set, it was never part of this extract at all -
    so (UREGBEZ, UKREIS) codes can't be resolved to official Gemeindeschlüssel
    or Bundesland/Kreis names. Deriving them from the nearest real accident
    location at least ties a map click to a code combination that genuinely
    co-occurs in the data, instead of asking the user to guess a raw pair.
    """
    try:
        con = duckdb.connect()
        query = f"""
            SELECT UREGBEZ, UKREIS
            FROM '{ACCIDENTS_PARQUET}'
            ORDER BY (LAT - {lat}) * (LAT - {lat}) + (LON - {lon}) * (LON - {lon})
            LIMIT 1
        """  # noqa: S608
        row = con.execute(query).fetchone()
        return str(row[0]), str(row[1])
    except Exception:
        logger.exception(f"Failed to find nearest UREGBEZ/UKREIS for ({lat}, {lon})")
        raise


def precision_decimals(precision: float) -> int:
    """Convert a grid precision like 0.1 or 0.5 into a ROUND() decimal-places argument."""
    if precision <= 0:
        raise ValueError(f"precision must be positive, got {precision}")
    return max(0, -int(round(math.log10(precision))))


@st.cache_data
def load_severity_grid(precision: float = 0.1) -> pd.DataFrame:
    """Aggregate accidents.parquet into a lat/lon grid with per-cell KSI/slight counts.

    UKATGEORIE in (1, 2) is KSI, UKATGEORIE == 3 is slight (verified against
    notebooks/01_Q_Phase.py lines 156-157; no invalid UKATGEORIE values exist
    in this file). Aggregation happens entirely in DuckDB so the ~2.09M-row
    parquet is never loaded into pandas row-by-row - only the grouped result
    (a few thousand rows at precision=0.1) crosses into pandas.
    """
    try:
        con = duckdb.connect()
        query = f"""
            SELECT
                ROUND(LAT, {precision_decimals(precision)}) AS lat_bin,
                ROUND(LON, {precision_decimals(precision)}) AS lon_bin,
                SUM(CASE WHEN UKATGEORIE IN (1, 2) THEN 1 ELSE 0 END) AS ksi_count,
                SUM(CASE WHEN UKATGEORIE = 3 THEN 1 ELSE 0 END) AS slight_count,
                COUNT(*) AS total
            FROM '{ACCIDENTS_PARQUET}'
            GROUP BY 1, 2
        """  # noqa: S608
        return con.execute(query).df()
    except Exception:
        logger.exception("Failed to load severity grid")
        raise


@st.cache_resource
def build_severity_map(precision: float = 0.1):
    """Build the folium severity map once and cache the Map object across reruns.

    Uses `folium.Circle` (radius in metres) rather than `CircleMarker` (radius
    in screen pixels) so a cell's marker stays the same size relative to the
    real geography at every zoom level, instead of shrinking to a sliver of a
    street once zoomed in. KSI-dominant and slight-dominant cells go into
    separate `FeatureGroup`s so `LayerControl` lets the user toggle each
    severity class on/off independently.

    folium.Map objects aren't relevant to compare by value, so this uses
    cache_resource (identity-cached singleton), not cache_data.
    """
    import folium

    grid_df = load_severity_grid(precision)
    severity_map = folium.Map(location=[51.1657, 10.4515], zoom_start=6)
    ksi_group = folium.FeatureGroup(name="KSI-dominant cells", show=True)
    slight_group = folium.FeatureGroup(name="Slight-dominant cells", show=True)
    for _, cell in grid_df.iterrows():
        ksi_share = cell["ksi_count"] / cell["total"]
        is_ksi_dominant = ksi_share >= 0.5
        color = SEVERITY_COLORS["KSI"] if is_ksi_dominant else SEVERITY_COLORS["slight"]
        target_group = ksi_group if is_ksi_dominant else slight_group
        folium.Circle(
            location=[cell["lat_bin"], cell["lon_bin"]],
            radius=min(5000, 300 + cell["total"] * 4),
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.5,
            popup=(
                f"KSI: {int(cell['ksi_count'])}, slight: {int(cell['slight_count'])}, "
                f"total: {int(cell['total'])}"
            ),
        ).add_to(target_group)
    ksi_group.add_to(severity_map)
    slight_group.add_to(severity_map)
    folium.LayerControl(collapsed=False).add_to(severity_map)
    return severity_map


def get_column_spec(contract: dict, name: str) -> dict:
    """Return the required_columns entry for one column name."""
    for col in contract["required_columns"]:
        if col["name"] == name:
            return col
    raise KeyError(f"Column {name!r} not found in inference contract required_columns")


@st.cache_resource
def load_champion_model() -> Pipeline:
    """Load the fitted champion pipeline referenced by the inference contract.

    The returned Pipeline already includes preprocessing (encoding, scaling),
    so callers only need to pass a raw-feature row matching required_columns.
    """
    contract = load_inference_contract()
    return joblib.load(contract["model_path"])


def build_input_row(widget_values: dict, contract: dict) -> pd.DataFrame:
    """Assemble a one-row DataFrame matching the contract's required_columns.

    IstGkfz's contract entry says dtype "object" with string categories
    ["False", "True"], which looks like it wants a string cast - but verified
    empirically against the real committed data/processed/a3_binary_best_model.joblib,
    IstGkfz sits in the fitted ColumnTransformer's passthrough group alongside
    its five Ist* sibling columns, which are all cast to float32 together. A
    real bool converts fine (True -> 1.0); the string "False" raises
    ValueError: could not convert string to float. The contract's recorded
    deployment_model_sha256 also doesn't match the actual committed joblib's
    sha256, so its dtype/categories metadata for this column appears to
    describe a different artifact than the one actually deployed. IstGkfz is
    therefore passed through unchanged here, exactly like its siblings.
    """
    row = {}
    for col in contract["required_columns"]:
        name = col["name"]
        if name not in widget_values:
            raise KeyError(
                f"Missing value for required column {name!r} - the predictor form did not "
                "supply this input (contract/widget schema drift)."
            )
        row[name] = widget_values[name]
    ordered_columns = [col["name"] for col in contract["required_columns"]]
    df = pd.DataFrame([row])[ordered_columns]
    # Force just the bool-valued columns to stay as real Python bool objects.
    # A whole-frame dtype=object cast preserves bool identity too, but it also
    # turns numeric columns into object arrays, which breaks the champion
    # pipeline's numpy-ufunc-based transforms (e.g. np.log1p on
    # dwd_station_dist_km) - verified empirically via the real committed
    # model. Scoping the object cast to only the bool columns keeps numeric
    # columns as native float/int dtypes for the pipeline while still letting
    # `row.loc[0, "IstPKW"] is True` hold, since plain pandas would otherwise
    # coerce a bool column to numpy.bool_.
    bool_cols = [name for name in ordered_columns if isinstance(row[name], bool)]
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(object)
    return df


def predict_ksi(model: Pipeline, row: pd.DataFrame, threshold: float) -> tuple[float, int]:
    """Predict KSI probability and thresholded label for one input row.

    Uses the contract's tuned decision threshold (0.4986), not sklearn's
    default 0.5 - the champion was selected and evaluated at this threshold.
    """
    proba = float(model.predict_proba(row)[0][1])
    prediction = int(proba >= threshold)
    return proba, prediction


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
