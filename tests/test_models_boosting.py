import numpy as np
import pandas as pd

from unfallatlas.features.preprocessing import build_preprocessor
from unfallatlas.models.boosting import (
    build_catboost_pipeline,
    build_lightgbm_pipeline,
    build_random_forest_pipeline,
    build_xgboost_pipeline,
    gpu_available,
)


def _toy_X_y(n=120):
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {
            "UMONAT": rng.integers(1, 13, n),
            "USTUNDE": rng.integers(0, 24, n),
            "UWOCHENTAG": rng.integers(1, 8, n),
            "UART": rng.integers(0, 10, n),
            "UTYP1": rng.integers(1, 8, n),
            "ULICHTVERH": rng.integers(0, 3, n),
            "STRZUSTAND": rng.integers(0, 3, n),
            "IstRad": rng.integers(0, 2, n),
            "IstPKW": rng.integers(0, 2, n),
            "IstFuss": rng.integers(0, 2, n),
            "IstKrad": rng.integers(0, 2, n),
            "IstGkfz": rng.integers(0, 2, n),
            "IstSonstig": rng.integers(0, 2, n),
            "LON": rng.uniform(6, 15, n),
            "LAT": rng.uniform(47, 55, n),
            "UREGBEZ": rng.integers(1, 5, n),
            "UKREIS": rng.integers(1000, 1050, n),
            "dwd_temp_air_2m": rng.normal(10, 5, n),
            "dwd_precip_mm": rng.exponential(1.0, n),
            "dwd_visibility_m": rng.exponential(5000, n),
            "dwd_wind_speed_ms": rng.normal(3, 1, n),
            "dwd_station_dist_km": rng.uniform(0.1, 40, n),
        }
    )
    y = pd.Series(rng.choice([1, 2, 3], n, p=[0.1, 0.3, 0.6]))
    return X, y


def test_random_forest_pipeline_predicts_known_labels():
    X, y = _toy_X_y()
    pipe = build_random_forest_pipeline(build_preprocessor(), class_weight="balanced")
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(preds).issubset({1, 2, 3})


def test_xgboost_pipeline_predicts_known_labels_not_zero_indexed():
    X, y = _toy_X_y()
    pipe = build_xgboost_pipeline(build_preprocessor())
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(preds).issubset({1, 2, 3})


def test_lightgbm_pipeline_predicts_known_labels():
    X, y = _toy_X_y()
    pipe = build_lightgbm_pipeline(build_preprocessor(), class_weight="balanced")
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(preds).issubset({1, 2, 3})


def test_catboost_pipeline_predicts_known_labels():
    X, y = _toy_X_y()
    pipe = build_catboost_pipeline(build_preprocessor())
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(np.asarray(preds).ravel()).issubset({1, 2, 3})


def test_xgboost_pipeline_does_not_require_zero_indexed_labels():
    """Regression test: XGBClassifier with an explicit num_class/objective
    raises ValueError on {1,2,3}-coded labels unless internally remapped to
    {0,1,2} and back. This must not resurface silently.
    """
    X, y = _toy_X_y()
    pipe = build_xgboost_pipeline(build_preprocessor())
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(preds) <= {1, 2, 3}
    assert 0 not in set(preds)


def test_gpu_available_returns_bool_and_is_consistent():
    result = gpu_available()
    assert isinstance(result, bool)
    assert gpu_available() == result  # lru_cache'd, must be stable within a process


def test_builders_accept_explicit_use_gpu_false_forcing_cpu():
    """use_gpu=False must always work, regardless of what's auto-detected on
    this machine — this is the portability guarantee for non-GPU machines."""
    X, y = _toy_X_y()
    preprocessor = build_preprocessor()

    rf_pipe = build_random_forest_pipeline(preprocessor)
    rf_pipe.fit(X, y)
    assert set(rf_pipe.predict(X)).issubset({1, 2, 3})

    xgb_pipe = build_xgboost_pipeline(preprocessor, use_gpu=False)
    xgb_pipe.fit(X, y)
    assert set(xgb_pipe.predict(X)).issubset({1, 2, 3})

    lgbm_pipe = build_lightgbm_pipeline(preprocessor, use_gpu=False)
    lgbm_pipe.fit(X, y)
    assert set(lgbm_pipe.predict(X)).issubset({1, 2, 3})

    cb_pipe = build_catboost_pipeline(preprocessor, use_gpu=False)
    cb_pipe.fit(X, y)
    assert set(np.asarray(cb_pipe.predict(X)).ravel()).issubset({1, 2, 3})
