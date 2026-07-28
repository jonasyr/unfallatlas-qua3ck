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
from typing import NamedTuple

import duckdb
import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

DATA_PROCESSED = Path("data/processed")
ACCIDENTS_PARQUET = Path("data/accidents.parquet")
ACCIDENTS_SPATIAL_PARQUET = Path("data/interim/accidents_with_weather_spatial.parquet")

DEFAULT_DWD_STATION_ID = "01975"
DEFAULT_H3_CELL = "881f15ad31fffff"
DEFAULT_DWD_STATION_DIST_KM = 9.51

SEVERITY_COLORS = {"KSI": "#E63946", "slight": "#2A9D8F"}

# Measured over the full committed data/accidents.parquet:
# 395766 KSI rows (UKATGEORIE IN (1, 2)) out of 2092401 total.
# Used only as a fallback if the DuckDB baseline query is unavailable; the live
# value comes from load_national_ksi_rate().
NATIONAL_KSI_RATE_FALLBACK = 0.1891444326398238

# Pseudo-count for shrinking a cell's KSI rate toward the national baseline, so a
# cell with 1 accident cannot register as "100% KSI". Chosen by measuring band
# populations over the real 4857-cell grid: k=10 still admits 6-accident cells into
# the top ">=2x" band (the noise shrinkage exists to suppress), while k=50
# over-shrinks it to 52 cells and flattens the map. k=20 fills all five bands
# (239 / 1137 / 2014 / 1263 / 204) with no band above 41% of cells.
SHRINKAGE_K = 20


class RiskBand(NamedTuple):
    """One relative-risk band: a half-open [lower, upper) ratio range and its color."""

    label: str
    lower: float
    upper: float
    color: str


# Diverging ramp anchored on the app's two existing brand colors
# (SEVERITY_COLORS["slight"] teal -> SEVERITY_COLORS["KSI"] red) with a warm sand
# midpoint. Deliberately hand-picked rather than RGB-interpolated: linear
# interpolation between teal and red passes through muddy brown-grey (#886B6A at
# the midpoint), which does not read as an ordered risk scale.
RISK_BANDS = (
    RiskBand("Well below average (<0.75x)", 0.0, 0.75, SEVERITY_COLORS["slight"]),
    RiskBand("Around average (0.75-1.1x)", 0.75, 1.1, "#8FC7BE"),
    RiskBand("Elevated (1.1-1.5x)", 1.1, 1.5, "#F2C879"),
    RiskBand("High (1.5-2x)", 1.5, 2.0, "#EE8062"),
    RiskBand("Very high (>=2x)", 2.0, float("inf"), SEVERITY_COLORS["KSI"]),
)

# Fill-opacity tiers by a cell's accident count, so a thinly-sampled cell reads as
# uncertain instead of looking as confident as a well-sampled one. Three discrete
# steps rather than a continuous fade, so the legend can state exactly what a pale
# cell means instead of implying unearned precision.
CONFIDENCE_OPACITY_TIERS = ((20, 0.25), (100, 0.45))
CONFIDENCE_OPACITY_MAX = 0.65


def shrunk_relative_risk(
    ksi_count: int, total: int, baseline: float, k: int = SHRINKAGE_K
) -> float:
    """Return a cell's KSI rate relative to the national baseline, shrunk toward it.

    An absolute "is KSI the local majority?" test is the wrong question: KSI is a
    ~18.9% minority outcome nationally, so a cell must reach 2.64x the national rate
    before crossing 50%. Only 123 of 4857 cells do, leaving 97.5% of the map a single
    color. Relative risk asks the answerable question - "is this cell worse than
    normal, and by how much?" - and the k-weighted shrinkage stops small cells from
    reaching extremes on noise.

    Returns 1.0 (exactly baseline) for an empty cell, since with no evidence the best
    estimate is the prior.
    """
    if baseline <= 0:
        raise ValueError(f"baseline must be positive, got {baseline}")
    shrunk_rate = (ksi_count + k * baseline) / (total + k)
    return shrunk_rate / baseline


def risk_band_index(relative_risk: float) -> int:
    """Return the index into RISK_BANDS for a relative-risk ratio.

    Bands are half-open [lower, upper), so a ratio landing exactly on a boundary
    falls into the higher band and every ratio maps to exactly one band.
    """
    if relative_risk < 0:
        raise ValueError(f"relative_risk must be non-negative, got {relative_risk}")
    for index, band in enumerate(RISK_BANDS):
        if relative_risk < band.upper:
            return index
    return len(RISK_BANDS) - 1


def confidence_opacity(total: int) -> float:
    """Return the fill opacity for a cell, lower when its sample is thin."""
    for threshold, opacity in CONFIDENCE_OPACITY_TIERS:
        if total < threshold:
            return opacity
    return CONFIDENCE_OPACITY_MAX


def severity_legend_markdown() -> str:
    """Render the static map legend as markdown with inline color swatches.

    folium's LayerControl names the bands but cannot show their colors or explain the
    opacity convention, so this legend carries both.
    """
    swatches = "\n".join(
        f'- <span style="display:inline-block;width:0.85rem;height:0.85rem;'
        f"background:{band.color};border:1px solid rgba(0,0,0,0.25);"
        f'vertical-align:middle;margin-right:0.5rem"></span>{band.label}'
        for band in RISK_BANDS
    )
    return (
        "**Relative KSI risk** - each cell's share of killed/seriously-injured "
        "accidents, compared against the national average of **18.9%**. "
        f"A cell at 2x is twice as likely to be severe as Germany overall.\n\n"
        f"{swatches}\n\n"
        "Rates are shrunk toward the national average in proportion to how few "
        "accidents a cell has, so a single severe accident cannot mark a cell as "
        "high-risk. Fill opacity shows that confidence directly: cells with fewer "
        "than 20 accidents are rendered faintest, under 100 mid-way, and 100 or "
        "more at full strength. Circle size still scales with the cell's total "
        "accident count."
    )


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

# Two real accident records (from data/interim/accidents_with_weather_spatial.parquet)
# picked to illustrate the champion model's behavior at the extremes, since it
# lands near the 0.4986 decision threshold most of the time on typical inputs.
# Found by batch-scoring a 20k-row sample through the actual committed
# a3_binary_best_model.joblib and taking the min/max predicted KSI probability -
# not hand-picked or invented. EXAMPLE_LOW_RISK scores ~0.03 (clearly "Slight"),
# EXAMPLE_HIGH_RISK scores ~0.89 (clearly "KSI"), both well clear of the
# threshold on the same pipeline used at prediction time.
EXAMPLE_LOW_RISK = {
    "UREGBEZ": "1",
    "UKREIS": "03",
    "UMONAT": 2,
    "USTUNDE": 15,
    "UWOCHENTAG": 6,
    "UART": 2,
    "UTYP1": 6,
    "ULICHTVERH": 0,
    "STRZUSTAND": 1,
    "IstRad": False,
    "IstPKW": True,
    "IstFuss": False,
    "IstKrad": False,
    "IstGkfz": False,
    "IstSonstig": False,
    "LON": 9.986046311,
    "LAT": 53.549888727,
    "dwd_station_id": "01975",
    "dwd_station_dist_km": 9.266605087571069,
    "dwd_temp_air_2m": 8.193103448275862,
    "dwd_precip_mm": 0.18275862068965518,
    "dwd_visibility_m": 26945.51724137931,
    "dwd_wind_speed_ms": 5.637931034482759,
    "_precip_bucket": "light (0–5 mm)",
    "h3_cell": "881f15ad31fffff",
    "osm_dominant_road_class": "primary",
    "osm_maxspeed_mean": 42.53699788583509,
    "osm_maxspeed_max": 50.0,
    "osm_road_density": 2470.0,
    "osm_way_count": 588.0,
}
EXAMPLE_HIGH_RISK = {
    "UREGBEZ": "5",
    "UKREIS": "23",
    "UMONAT": 5,
    "USTUNDE": 16,
    "UWOCHENTAG": 3,
    "UART": 8,
    "UTYP1": 1,
    "ULICHTVERH": 0,
    "STRZUSTAND": 0,
    "IstRad": False,
    "IstPKW": False,
    "IstFuss": False,
    "IstKrad": True,
    "IstGkfz": False,
    "IstSonstig": False,
    "LON": 12.45280067,
    "LAT": 50.453367958,
    "dwd_station_id": "00840",
    "dwd_station_dist_km": 17.805339919603124,
    "dwd_temp_air_2m": 14.674193548387096,
    "dwd_precip_mm": 0.09999999999999999,
    "dwd_visibility_m": 47991.6129032258,
    "dwd_wind_speed_ms": 2.7516129032258063,
    "_precip_bucket": "light (0–5 mm)",
    "h3_cell": "881f15ad31fffff",
    "osm_dominant_road_class": "secondary",
    "osm_maxspeed_mean": 100.0,
    "osm_maxspeed_max": 100.0,
    "osm_road_density": 130.0,
    "osm_way_count": 14.0,
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
def nearest_location_features(lat: float, lon: float) -> dict:
    """Look up every autofillable feature for the accident record nearest a map click.

    This dataset has no ULAND (Bundesland) column anywhere - not just excluded
    from the model's feature set, it was never part of this extract at all -
    so (UREGBEZ, UKREIS) codes can't be resolved to official Gemeindeschlüssel
    or Bundesland/Kreis names. Deriving them, and the road/weather context
    columns, from the nearest real accident location at least ties a map
    click to values that genuinely co-occur in the data, instead of asking
    the user to guess a raw combination for every widget.

    Reads data/interim/accidents_with_weather_spatial.parquet (not the plain
    accidents.parquet - that one only has the raw Unfallatlas columns, none
    of the osm_*/dwd_weather columns needed here). A ~1-degree bounding-box
    pre-filter keeps the nearest-neighbor ORDER BY cheap; if a click lands
    somewhere with no rows inside that box (e.g. right at the edge of the
    covered area), it retries once without the filter.
    """
    columns = [
        "UREGBEZ",
        "UKREIS",
        "dwd_station_id",
        "dwd_station_dist_km",
        "dwd_temp_air_2m",
        "dwd_precip_mm",
        "dwd_visibility_m",
        "dwd_wind_speed_ms",
        "osm_dominant_road_class",
        "osm_maxspeed_mean",
        "osm_maxspeed_max",
        "osm_road_density",
        "osm_way_count",
        "h3_cell",
    ]
    try:
        con = duckdb.connect()

        def _query(where: str) -> tuple | None:
            select_list = ", ".join(columns)
            query = f"""
                SELECT {select_list}
                FROM '{ACCIDENTS_SPATIAL_PARQUET}'
                {where}
                ORDER BY (LAT - {lat}) * (LAT - {lat}) + (LON - {lon}) * (LON - {lon})
                LIMIT 1
            """  # noqa: S608
            return con.execute(query).fetchone()

        row = _query(
            f"WHERE LAT BETWEEN {lat - 0.5} AND {lat + 0.5} "
            f"AND LON BETWEEN {lon - 0.5} AND {lon + 0.5}"
        )
        if row is None:
            row = _query("")
        return dict(zip(columns, row, strict=True))
    except Exception:
        logger.exception(f"Failed to find nearest location features for ({lat}, {lon})")
        raise


def precision_decimals(precision: float) -> int:
    """Convert a grid precision like 0.1 or 0.5 into a ROUND() decimal-places argument."""
    if precision <= 0:
        raise ValueError(f"precision must be positive, got {precision}")
    return max(0, -int(round(math.log10(precision))))


@st.cache_data
def load_national_ksi_rate() -> float:
    """Return the share of all accidents that are KSI, across the whole dataset.

    This is the baseline the Overview map's relative-risk bands are measured
    against. Computed live from the committed parquet rather than hardcoded, so it
    stays correct if the dataset is ever revised.
    """
    try:
        con = duckdb.connect()
        query = f"""
            SELECT AVG(CASE WHEN UKATGEORIE IN (1, 2) THEN 1.0 ELSE 0.0 END)
            FROM '{ACCIDENTS_PARQUET}'
        """  # noqa: S608
        return float(con.execute(query).fetchone()[0])
    except Exception:
        logger.exception("Failed to load national KSI rate")
        raise


@st.cache_data
def load_severity_grid(precision: float = 0.1) -> pd.DataFrame:
    """Aggregate accidents.parquet into a lat/lon grid with per-cell KSI/slight counts.

    UKATGEORIE in (1, 2) is KSI, UKATGEORIE == 3 is slight (verified against
    notebooks/01_Q_Phase.py lines 156-157; no invalid UKATGEORIE values exist
    in this file). Aggregation happens entirely in DuckDB so the ~2.09M-row
    parquet is never loaded into pandas row-by-row - only the grouped result
    (a few thousand rows at precision=0.1) crosses into pandas.

    `lat_bin`/`lon_bin` are the grouping key only. `center_lat`/`center_lon` are the
    mean coordinates of the accidents actually inside the cell, and are what callers
    should plot: the rounded bin coordinate sits a mean of 1.71 km (p95 3.80 km, max
    6.48 km) away from the cell's own accidents, because accidents are not uniformly
    distributed inside a ~11 km cell.
    """
    try:
        con = duckdb.connect()
        query = f"""
            SELECT
                ROUND(LAT, {precision_decimals(precision)}) AS lat_bin,
                ROUND(LON, {precision_decimals(precision)}) AS lon_bin,
                AVG(LAT) AS center_lat,
                AVG(LON) AS center_lon,
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
def build_severity_base_map():
    """Return an empty, cached folium map centred on Germany.

    Deliberately carries NO layers. Every marker layer reaches the browser through
    `st_folium(feature_group_to_add=...)` instead. An earlier version attached
    FeatureGroups and a LayerControl here with `.add_to()`, which raised
    `ReferenceError: feature_group_<hash> is not defined` in the browser and blanked
    the entire map (AppTest saw no exception at all - only a real browser catches
    this). streamlit-folium re-injects the rendered map into its own `map_div`
    execution context and only rewrites layer references for objects passed through
    its own parameters, so layers baked in here do not resolve.

    folium.Map objects aren't meaningfully comparable by value, so this uses
    cache_resource (identity-cached singleton), not cache_data.
    """
    import folium

    return folium.Map(location=[51.1657, 10.4515], zoom_start=6)


@st.cache_resource
def build_severity_feature_groups(precision: float = 0.1):
    """Build one toggleable FeatureGroup per relative-risk band.

    Returned as a plain list for `st_folium(feature_group_to_add=...)`; the caller
    must not attach these to a map (see build_severity_base_map). folium's
    LayerControl over these groups doubles as the map's legend and lets a reader
    isolate a single band - for example showing only ">=2x average" cells to see
    where they cluster.

    Uses `folium.Circle` (radius in metres) rather than `CircleMarker` (radius in
    screen pixels) so a cell's marker keeps a constant size relative to the real
    geography at every zoom level. Cells are drawn at their accident centroid, not
    at the rounded grid-bin coordinate.

    Cached because building ~4857 Circle objects costs roughly 3.4 s and 4.9 MB of
    HTML, which is too slow to repeat on every rerun.
    """
    import folium

    grid_df = load_severity_grid(precision)
    baseline = load_national_ksi_rate()
    groups = [folium.FeatureGroup(name=band.label, show=True) for band in RISK_BANDS]

    for cell in grid_df.itertuples():
        relative_risk = shrunk_relative_risk(cell.ksi_count, cell.total, baseline)
        band_index = risk_band_index(relative_risk)
        band = RISK_BANDS[band_index]
        folium.Circle(
            location=[cell.center_lat, cell.center_lon],
            radius=min(5000, 300 + cell.total * 4),
            color=band.color,
            weight=1,
            fill=True,
            fill_color=band.color,
            fill_opacity=confidence_opacity(cell.total),
            popup=(
                f"{relative_risk:.2f}x national KSI rate ({band.label})<br>"
                f"KSI: {int(cell.ksi_count)}, slight: {int(cell.slight_count)}, "
                f"total: {int(cell.total)}"
            ),
        ).add_to(groups[band_index])

    return groups


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
