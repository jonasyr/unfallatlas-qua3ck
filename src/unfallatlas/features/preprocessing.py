"""U-phase §10 preprocessing contract, implemented as a single sklearn Pipeline.

The U-phase decided; this module implements exactly what
notebooks/02_U_Phase.py §10 specifies — no additional engineered features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from unfallatlas.features.temporal import cyclic_encode


def _log_with_offset(values: np.ndarray) -> np.ndarray:
    """log(x + 1e-6) for dwd_station_dist_km — a module-level function so the
    fitted preprocessor (and any Pipeline containing it) can be pickled by
    joblib. A lambda here would raise PicklingError on joblib.dump()."""
    return np.log(values + 1e-6)


TARGET_COLUMN = "UKATGEORIE"
SPLIT_YEAR_COLUMN = "UJAHR"
NON_FEATURE_COLUMNS = ["OBJECTID", "UGEMEINDE"]

CYCLIC_COLUMNS = {"UMONAT": 12, "USTUNDE": 24, "UWOCHENTAG": 7}
ONEHOT_COLUMNS = ["UART", "UTYP1", "ULICHTVERH", "STRZUSTAND"]
TARGET_ENCODED_COLUMNS = ["UREGBEZ", "UKREIS"]
PASSTHROUGH_COLUMNS = [
    "IstRad",
    "IstPKW",
    "IstFuss",
    "IstKrad",
    "IstGkfz",
    "IstSonstig",
    "LON",
    "LAT",
]
LOG1P_COLUMNS = ["dwd_precip_mm", "dwd_visibility_m"]
LOG_COLUMNS = ["dwd_station_dist_km"]
PLAIN_NUMERIC_COLUMNS = ["dwd_temp_air_2m", "dwd_wind_speed_ms"]


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """Mean-target encoding with additive smoothing (U-phase §10).

    Encodes each category as the smoothed mean of the *numeric* target code
    (1/2/3) observed for that category in the training fold — consistent
    with the Q-phase §5 note that the three classes have a natural order.
    Fit on training data only; unseen categories at transform time receive
    the global training mean.
    """

    def __init__(self, columns: list[str], smoothing: float = 10.0):
        self.columns = columns
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        y_numeric = pd.Series(np.asarray(y), index=X.index, dtype=float)
        self.global_mean_ = float(y_numeric.mean())
        self.mappings_: dict[str, pd.Series] = {}
        for col in self.columns:
            grp = y_numeric.groupby(X[col], observed=True)
            counts = grp.count()
            means = grp.mean()
            self.mappings_[col] = (counts * means + self.smoothing * self.global_mean_) / (
                counts + self.smoothing
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in self.columns:
            out[f"{col}_target_enc"] = (
                X[col].map(self.mappings_[col]).fillna(self.global_mean_).astype(float)
            )
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{col}_target_enc" for col in self.columns])


class CyclicEncoder(BaseEstimator, TransformerMixin):
    """sklearn-compatible wrapper around ``cyclic_encode`` for a ColumnTransformer."""

    def __init__(self, period: int):
        self.period = period

    def fit(self, X: pd.DataFrame, y=None):
        self.column_ = X.columns[0]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        encoded = cyclic_encode(X, self.column_, self.period)
        return encoded[[f"{self.column_}_sin", f"{self.column_}_cos"]].to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.column_}_sin", f"{self.column_}_cos"])


def build_preprocessor(scale_for_linear: bool = False) -> ColumnTransformer:
    """Build the ColumnTransformer implementing U-phase §10 verbatim.

    scale_for_linear=True additionally scales LON/LAT and the binary
    transport-mode flags — required by the Logistic Regression baseline;
    U-phase §10 marks this scaling "only for distance-based baselines,"
    which tree models do not need.
    """
    transformers = [
        (f"cyclic_{col}", CyclicEncoder(period=period), [col])
        for col, period in CYCLIC_COLUMNS.items()
    ]

    transformers.append(
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ONEHOT_COLUMNS)
    )
    transformers.append(
        ("target_enc", TargetMeanEncoder(columns=TARGET_ENCODED_COLUMNS), TARGET_ENCODED_COLUMNS)
    )

    log1p_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    transformers.append(("log1p_cols", log1p_pipeline, LOG1P_COLUMNS))

    log_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "log",
                FunctionTransformer(_log_with_offset, feature_names_out="one-to-one"),
            ),
        ]
    )
    transformers.append(("log_cols", log_pipeline, LOG_COLUMNS))

    plain_numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    transformers.append(("plain_numeric", plain_numeric_pipeline, PLAIN_NUMERIC_COLUMNS))

    if scale_for_linear:
        passthrough_scaled_pipeline = Pipeline(
            steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        transformers.append(
            ("passthrough_scaled", passthrough_scaled_pipeline, PASSTHROUGH_COLUMNS)
        )
    else:
        transformers.append(("passthrough", "passthrough", PASSTHROUGH_COLUMNS))

    return ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=False
    )


def load_training_frame(base_dir: Path) -> pd.DataFrame:
    """Load the DWD-and-OSM-enriched accidents frame built by the U-phase.

    Reuses the cache from unfallatlas.data.dwd.build_weather_features and
    unfallatlas.data.osm.build_spatial_features (A³ does not rebuild this
    cache — raises if it is missing).
    """
    cache = base_dir / "data" / "interim" / "accidents_with_weather_spatial.parquet"
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} not found. Run notebooks/02_U_Phase.ipynb §8.5 (weather) "
            "and §8.x (OSM spatial features) first."
        )
    df = pd.read_parquet(cache)
    return df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])


def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train 2016-2022 / val 2023 / test 2024 — Q-phase §6, U-phase §9.2."""
    train = df[df[SPLIT_YEAR_COLUMN] <= 2022].reset_index(drop=True)
    val = df[df[SPLIT_YEAR_COLUMN] == 2023].reset_index(drop=True)
    test = df[df[SPLIT_YEAR_COLUMN] == 2024].reset_index(drop=True)
    return train, val, test


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop UJAHR/target from the feature frame; return (X, y)."""
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN, SPLIT_YEAR_COLUMN])
    return X, y
