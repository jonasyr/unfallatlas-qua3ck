import numpy as np
import pandas as pd
from sklearn.base import clone

from unfallatlas.features.preprocessing import build_preprocessor
from unfallatlas.models.boosting import (
    build_catboost_pipeline,
    build_lightgbm_binary_pipeline,
    build_lightgbm_pipeline,
    build_random_forest_pipeline,
    build_xgboost_binary_pipeline,
    build_xgboost_pipeline,
    gpu_available,
)
from unfallatlas.models.imbalance import balanced_sample_weight


def _toy_X_y(n=120):
    rng = np.random.default_rng(1)
    road_classes = np.array(["primary", "secondary", "residential", "motorway", "tertiary", None])
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
            "osm_dominant_road_class": rng.choice(road_classes, n),
            "osm_maxspeed_mean": rng.choice(
                [30.0, 50.0, 70.0, 100.0, np.nan], n, p=[0.2, 0.3, 0.2, 0.2, 0.1]
            ),
            "osm_maxspeed_max": rng.choice(
                [50.0, 70.0, 100.0, 130.0, np.nan], n, p=[0.2, 0.3, 0.2, 0.2, 0.1]
            ),
            "osm_road_density": rng.choice([*rng.exponential(500, n // 2), np.nan], n),
            "osm_way_count": rng.choice([*rng.integers(2, 400, n // 2), np.nan], n),
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


def test_catboost_pipeline_has_no_class_weights_param():
    """Regression test: build_catboost_pipeline must never expose a
    class_weights constructor kwarg again. Confirmed empirically that
    sklearn.base.clone() unconditionally fails on any CatBoostClassifier
    configured with a non-None class_weights (list or dict, fitted or
    not) - every sklearn CV utility (cross_val_score, cross_validate,
    GridSearchCV, ...) clones the estimator internally per fold, so this
    silently breaks Optuna/cross-validation the moment class_weights is
    reintroduced anywhere in this builder.
    """
    pipe = build_catboost_pipeline(build_preprocessor())
    assert "class_weights" not in pipe.named_steps["classify"].get_params()


def test_catboost_pipeline_clones_successfully():
    """Regression test for the clone()/class_weights incompatibility:
    a CatBoost pipeline built by this function must always survive
    sklearn.base.clone(), since Optuna's per-trial objective and every
    sklearn CV utility clone the estimator internally per fold."""
    pipe = build_catboost_pipeline(build_preprocessor())
    clone(pipe)  # must not raise RuntimeError


def test_catboost_pipeline_supports_sample_weight_at_fit_time():
    """The class-weighted configuration is applied via sample_weight= at
    fit time (mirroring build_xgboost_pipeline's pattern), not via a
    constructor kwarg - confirm the Pipeline actually routes
    classify__sample_weight through to CatBoostClassifier.fit()."""
    X, y = _toy_X_y()
    weights = balanced_sample_weight(y)
    pipe = build_catboost_pipeline(build_preprocessor())
    pipe.fit(X, y, classify__sample_weight=weights)
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


def test_lightgbm_defaults_to_cpu_without_opencl_auto_detection():
    """A CUDA GPU alone does not prove LightGBM can use its OpenCL backend."""
    lgbm = build_lightgbm_pipeline(build_preprocessor()).named_steps["classify"]
    assert lgbm.get_params()["device"] == "cpu"


def test_build_lightgbm_binary_pipeline_fits_and_predicts_binary():
    X, y3 = _toy_X_y(n=120)
    # Binary target: 1 if original label == 1 or 2, else 0
    y_bin = (np.array(y3) <= 2).astype(int)
    preprocessor = build_preprocessor()
    pipeline = build_lightgbm_binary_pipeline(preprocessor)
    pipeline.fit(X, y_bin)
    preds = pipeline.predict(X)
    assert set(np.unique(preds)) <= {0, 1}
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_build_xgboost_binary_pipeline_fits_and_predicts_binary():
    """Regression test for a real bug found via live execution: reusing
    build_xgboost_pipeline (hardcoded objective='multi:softprob', num_class=3)
    for a binary {0,1} target makes _ZeroIndexedXGBClassifier.predict() raise
    IndexError, because argmax over 3 phantom classes can return index 2,
    which is out of range for a 2-element classes_ array. This builder must
    fit and predict binary labels without that wrapper or that objective."""
    X, y3 = _toy_X_y(n=120)
    y_bin = (np.array(y3) <= 2).astype(int)
    preprocessor = build_preprocessor()
    pipeline = build_xgboost_binary_pipeline(preprocessor)
    pipeline.fit(X, y_bin)
    preds = pipeline.predict(X)
    assert set(np.unique(preds)) <= {0, 1}
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_build_lightgbm_binary_pipeline_set_params_works():
    X, y3 = _toy_X_y(n=60)
    y_bin = (np.array(y3) <= 2).astype(int)
    preprocessor = build_preprocessor()
    pipeline = build_lightgbm_binary_pipeline(preprocessor)
    pipeline.set_params(classify__n_estimators=50)
    pipeline.fit(X, y_bin)
    assert pipeline.named_steps["classify"].n_estimators == 50


def test_builders_accept_explicit_use_gpu_false_forcing_cpu():
    """use_gpu=False must always force CPU, regardless of local hardware."""
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
