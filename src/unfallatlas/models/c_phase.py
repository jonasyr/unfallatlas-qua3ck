"""C-phase (Conclude & Compare) analysis helpers.

Pure functions consumed by notebooks/04_C_Phase.ipynb. No notebook-specific
state; every function takes explicit DataFrames/Series and returns a
DataFrame or JSON-serializable dict.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

# Human-readable labels for the coded categoricals shown in C-phase error-slice
# and SHAP visualizations. Mirrors notebooks/02_U_Phase.py's "Human-readable
# labels" section (source: Datensatzbeschreibung Unfallatlas, Stand
# 10.06.2025) - duplicated rather than imported so the notebook-side module
# stays self-contained, but the label text is the same codebook lookup.
# Visualizations must decode these inline (never show a bare code like
# "UART=2" next to a separate legend the reader has to cross-reference).
ULICHTVERH_LABELS = {0: "Tageslicht", 1: "Dämmerung", 2: "Dunkelheit"}
STRZUSTAND_LABELS = {0: "Trocken", 1: "Nass/feucht/schlüpfrig", 2: "Winterglatt"}
UART_LABELS = {
    0: "Anderer Art",
    1: "Zzs. ruhendes Fz.",
    2: "Zzs. vorausfahrendes Fz. (Auffahrunfall)",
    3: "Zzs. seitlich gleichfahrendes Fz.",
    4: "Zzs. Gegenverkehr",
    5: "Zzs. einbiegendes/kreuzendes Fz.",
    6: "Zzs. Fußgänger",
    7: "Aufprall Fahrbahnhindernis",
    8: "Abkommen nach rechts",
    9: "Abkommen nach links",
}
UTYP1_LABELS = {
    1: "Fahrunfall",
    2: "Abbiegeunfall",
    3: "Einbiegen/Kreuzen",
    4: "Überschreiten",
    5: "Ruhender Verkehr",
    6: "Längsverkehr",
    7: "Sonstiger Unfall",
}
COL_CODE_LABELS = {
    "ULICHTVERH": ULICHTVERH_LABELS,
    "STRZUSTAND": STRZUSTAND_LABELS,
    "UART": UART_LABELS,
    "UTYP1": UTYP1_LABELS,
}
COL_DISPLAY_NAMES = {
    "UART": "Unfallart",
    "UTYP1": "Unfalltyp",
    "ULICHTVERH": "Lichtverhältnisse",
    "STRZUSTAND": "Straßenzustand",
    "osm_dominant_road_class": "Straßenklasse (OSM)",
    "_precip_bucket": "Niederschlag",
    "USTUNDE": "Uhrzeit",
    "UMONAT": "Monat",
    "UWOCHENTAG": "Wochentag",
    "UKREIS": "Kreis",
    "UREGBEZ": "Regierungsbezirk",
}

# Labels for the remaining feature columns the champion pipeline consumes
# (src/unfallatlas/features/preprocessing.py PASSTHROUGH_COLUMNS/LOG1P_COLUMNS/
# LOG_COLUMNS/PLAIN_NUMERIC_COLUMNS): the participant-type flags (Ist*) mirror
# docs/dataset/DSB_Unfallatlas.md "Beteiligungen"; the dwd_*/osm_* labels
# mirror notebooks/02_U_Phase.py's DWD_COL_LABELS plus the OSM road-context
# columns C-phase adds on top of U-Phase's dictionary.
FEATURE_LABELS = {
    "IstRad": "Fahrradbeteiligung",
    "IstPKW": "Pkw-Beteiligung",
    "IstFuss": "Fußgängerbeteiligung",
    "IstKrad": "Kraftradbeteiligung",
    "IstGkfz": "Güterkraftfahrzeug-Beteiligung",
    "IstSonstig": "Beteiligung sonstiges Verkehrsmittel",
    "LON": "Längengrad",
    "LAT": "Breitengrad",
    "dwd_temp_air_2m": "Lufttemperatur (°C)",
    "dwd_precip_mm": "Niederschlag (mm)",
    "dwd_visibility_m": "Sichtweite (m)",
    "dwd_wind_speed_ms": "Windgeschwindigkeit (m/s)",
    "dwd_station_dist_km": "Entfernung zur DWD-Station (km)",
    "osm_road_density": "Straßendichte (OSM)",
    "osm_way_count": "Straßenanzahl (OSM)",
    "osm_maxspeed_mean": "Höchstgeschwindigkeit, Mittel (OSM)",
    "osm_maxspeed_max": "Höchstgeschwindigkeit, Maximum (OSM)",
}


def decode_slice_label(column: str, value: object) -> str:
    """Human-readable "Spaltenname: Wertlabel" for one error-slice row.

    Used for chart axis labels and table display so a reader never has to
    cross-reference a raw code (e.g. "UART=2") against a separate legend.
    """
    display_name = COL_DISPLAY_NAMES.get(column, column)
    codes = COL_CODE_LABELS.get(column)
    if codes is not None:
        label = codes.get(int(value), str(value))
    elif column == "USTUNDE":
        label = f"{int(value):02d}:00 Uhr"
    else:
        label = str(value)
    return f"{display_name}: {label}"


_OSM_ROAD_CLASS_PREFIX = "osm_dominant_road_class_"
_TARGET_ENC_SUFFIX = "_target_enc"


def humanize_feature_name(name: str) -> str:
    """Decode a fitted-preprocessor output feature name for display.

    Covers every shape `ColumnTransformer.get_feature_names_out()` emits for
    the champion pipeline (src/unfallatlas/features/preprocessing.py):
    one-hot dummies for UART/UTYP1/ULICHTVERH/STRZUSTAND (e.g. "UART_2") and
    for osm_dominant_road_class (e.g. "osm_dominant_road_class_residential"),
    target-encoded columns ("UKREIS_target_enc"), cyclic sin/cos pairs
    ("USTUNDE_sin"), and the plain passthrough/DWD/OSM numeric columns via
    FEATURE_LABELS. Never leaves a raw code or column name for the reader to
    look up separately.
    """
    for col, codes in COL_CODE_LABELS.items():
        prefix = f"{col}_"
        if name.startswith(prefix):
            raw_value = name[len(prefix) :]
            label = codes.get(int(raw_value), raw_value)
            return f"{COL_DISPLAY_NAMES.get(col, col)}: {label}"
    if name.startswith(_OSM_ROAD_CLASS_PREFIX):
        value = name[len(_OSM_ROAD_CLASS_PREFIX) :]
        return f"{COL_DISPLAY_NAMES['osm_dominant_road_class']}: {value}"
    if name.endswith(_TARGET_ENC_SUFFIX):
        col = name[: -len(_TARGET_ENC_SUFFIX)]
        return f"{COL_DISPLAY_NAMES.get(col, col)} (zielcodiert)"
    if name.endswith(("_sin", "_cos")):
        col, component = name.rsplit("_", 1)
        return f"{COL_DISPLAY_NAMES.get(col, col)} (zyklisch, {component})"
    return FEATURE_LABELS.get(name, name)


QUALITATIVE_MATRIX_WEIGHTS = {
    "macro_f1": 0.30,
    "recall_ksi": 0.30,
    "latency_ms_per_1k": 0.10,
    "interpretability_score": 0.10,
    "robustness_score": 0.10,
    "training_cost_score": 0.10,
}

# Columns where a HIGHER raw value is WORSE (cost-type) and must be inverted
# before weighting: latency and training cost. All other columns are
# benefit-type (higher raw value is better) and used as-is.
_COST_TYPE_COLUMNS = {"latency_ms_per_1k", "training_cost_score"}


def compute_error_slices(
    y_true: pd.Series,
    y_pred: pd.Series,
    slice_frame: pd.DataFrame,
    slice_columns: list[str],
) -> pd.DataFrame:
    """False-negative / false-positive rate broken down by each slice column.

    One output row per (slice_column, slice_value). Rates are computed over
    all rows carrying that slice value, not only the errors, so they are
    directly comparable across slices of different sizes.
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    slice_frame = slice_frame.reset_index(drop=True)

    is_fn = (y_true == 1) & (y_pred == 0)
    is_fp = (y_true == 0) & (y_pred == 1)

    rows = []
    for col in slice_columns:
        values = slice_frame[col]
        for value, idx in values.groupby(values).groups.items():
            n = len(idx)
            n_fn = int(is_fn.loc[idx].sum())
            n_fp = int(is_fp.loc[idx].sum())
            n_actual_positive = int((y_true.loc[idx] == 1).sum())
            n_actual_negative = int((y_true.loc[idx] == 0).sum())
            rows.append(
                {
                    "slice_column": col,
                    "slice_value": value,
                    "n": n,
                    "n_false_negative": n_fn,
                    "n_false_positive": n_fp,
                    "false_negative_rate": (n_fn / n_actual_positive)
                    if n_actual_positive
                    else np.nan,
                    "false_positive_rate": (n_fp / n_actual_negative)
                    if n_actual_negative
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_qualitative_matrix(rows: list[dict]) -> pd.DataFrame:
    """Weighted multi-criteria comparison table, sorted best-first."""
    df = pd.DataFrame(rows).set_index("model")

    normalized = pd.DataFrame(index=df.index)
    for col, weight in QUALITATIVE_MATRIX_WEIGHTS.items():
        col_min, col_max = df[col].min(), df[col].max()
        span = (col_max - col_min) or 1.0
        scaled = (df[col] - col_min) / span
        if col in _COST_TYPE_COLUMNS:
            scaled = 1.0 - scaled
        normalized[col] = scaled * weight

    df["weighted_score"] = normalized.sum(axis=1)
    return df.reset_index().sort_values("weighted_score", ascending=False).reset_index(drop=True)


def _infer_source(col: str) -> str:
    """Best-effort provenance tag for a feature column.

    Based on the U-phase naming convention documented in docs/GLOSSARY.md
    (osm_*/h3_cell = OSM road-context enrichment, dwd_*/_precip_bucket = DWD
    weather enrichment) - not an assumption, a lookup against the same
    prefixes those enrichment steps actually use.
    """
    if col.startswith("dwd_") or col == "_precip_bucket":
        return "DWD weather enrichment (U-phase)"
    if col.startswith("osm_") or col == "h3_cell":
        return "OSM road-context enrichment (U-phase)"
    return "Unfallatlas raw/engineered (U-phase)"


def _infer_range(series: pd.Series) -> dict:
    """Valid range (numeric) or category list (categorical), from real data.

    Never assumed: derived from the actual observed column values so the
    K-phase implementer gets real validation bounds, not guesses. High-
    cardinality columns (e.g. district codes, station IDs) get an explicit
    note instead of an unusably long category list.
    """
    if pd.api.types.is_bool_dtype(series):
        return {"categories": [True, False]}
    if pd.api.types.is_numeric_dtype(series):
        return {"min": float(series.min()), "max": float(series.max())}
    n_unique = int(series.nunique(dropna=True))
    if n_unique <= 50:
        return {"categories": sorted(str(v) for v in series.dropna().unique())}
    return {"note": f"high-cardinality ({n_unique} unique values), no fixed category list"}


def build_inference_contract(
    feature_columns: list[str],
    dtypes: dict[str, str],
    model_card: dict,
    feature_frame: pd.DataFrame,
) -> dict:
    """JSON-serializable contract describing the champion model's input schema.

    `feature_frame` supplies the real observed range/categories per column
    (e.g. the training split) so the K-phase implementer never has to
    re-derive validation bounds or provenance from the notebooks.
    """
    required_columns = []
    for col in feature_columns:
        entry = {
            "name": col,
            "dtype": dtypes.get(col, "unknown"),
            "source": _infer_source(col),
        }
        entry.update(_infer_range(feature_frame[col]))
        required_columns.append(entry)

    return {
        "required_columns": required_columns,
        "threshold": model_card["optimal_threshold_val_2023"],
        "target_encoding": model_card["target_encoding"],
        "model_path": "data/processed/a3_binary_best_model.joblib",
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
