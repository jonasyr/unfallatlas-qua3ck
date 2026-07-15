import numpy as np
import pandas as pd

from unfallatlas.features.preprocessing import build_preprocessor
from unfallatlas.models.svm import (
    build_linear_svm_binary_pipeline,
    build_rbf_svm_binary_pipeline,
    build_sgd_hinge_binary_pipeline,
)


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
    y_bin = pd.Series(rng.choice([0, 1], n, p=[0.8, 0.2]))
    return X, y_bin


def test_linear_svm_binary_pipeline_predicts_binary_labels():
    X, y = _toy_X_y()
    pipe = build_linear_svm_binary_pipeline(build_preprocessor(scale_for_linear=True))
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(np.unique(preds)) <= {0, 1}


def test_linear_svm_binary_pipeline_exposes_decision_function():
    X, y = _toy_X_y()
    pipe = build_linear_svm_binary_pipeline(build_preprocessor(scale_for_linear=True))
    pipe.fit(X, y)
    scores = pipe.decision_function(X)
    assert scores.shape == (len(X),)


def test_sgd_hinge_binary_pipeline_predicts_binary_labels():
    X, y = _toy_X_y()
    pipe = build_sgd_hinge_binary_pipeline(build_preprocessor(scale_for_linear=True))
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(np.unique(preds)) <= {0, 1}


def test_sgd_hinge_binary_pipeline_uses_hinge_loss():
    """Regression test: this must stay a genuine (linear) SVM, not silently
    drift to log-loss logistic regression if someone edits the default."""
    pipe = build_sgd_hinge_binary_pipeline(build_preprocessor(scale_for_linear=True))
    assert pipe.named_steps["classify"].get_params()["loss"] == "hinge"


def test_rbf_svm_binary_pipeline_predicts_binary_labels():
    X, y = _toy_X_y()
    pipe = build_rbf_svm_binary_pipeline(build_preprocessor(scale_for_linear=True))
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(np.unique(preds)) <= {0, 1}


def test_rbf_svm_binary_pipeline_exposes_decision_function():
    X, y = _toy_X_y()
    pipe = build_rbf_svm_binary_pipeline(build_preprocessor(scale_for_linear=True))
    pipe.fit(X, y)
    scores = pipe.decision_function(X)
    assert scores.shape == (len(X),)


def test_svm_pipelines_accept_hyperparameter_overrides():
    X, y = _toy_X_y()
    linear_pipe = build_linear_svm_binary_pipeline(
        build_preprocessor(scale_for_linear=True), C=0.01
    )
    assert linear_pipe.named_steps["classify"].get_params()["C"] == 0.01

    sgd_pipe = build_sgd_hinge_binary_pipeline(
        build_preprocessor(scale_for_linear=True), alpha=1e-3
    )
    assert sgd_pipe.named_steps["classify"].get_params()["alpha"] == 1e-3

    rbf_pipe = build_rbf_svm_binary_pipeline(
        build_preprocessor(scale_for_linear=True), C=5.0, gamma=0.1
    )
    params = rbf_pipe.named_steps["classify"].get_params()
    assert params["C"] == 5.0
    assert params["gamma"] == 0.1
