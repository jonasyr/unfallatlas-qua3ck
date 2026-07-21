"""C-phase (Conclude & Compare) analysis helpers.

Pure functions consumed by notebooks/04_C_Phase.ipynb. No notebook-specific
state; every function takes explicit DataFrames/Series and returns a
DataFrame or JSON-serializable dict.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

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


def build_inference_contract(
    feature_columns: list[str],
    dtypes: dict[str, str],
    model_card: dict,
) -> dict:
    """JSON-serializable contract describing the champion model's input schema."""
    return {
        "required_columns": [
            {"name": col, "dtype": dtypes.get(col, "unknown")} for col in feature_columns
        ],
        "threshold": model_card["optimal_threshold_val_2023"],
        "target_encoding": model_card["target_encoding"],
        "model_path": "data/processed/a3_binary_best_model.joblib",
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
