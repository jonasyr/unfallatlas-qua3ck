import numpy as np
import pandas as pd

from unfallatlas.features.preprocessing import build_preprocessor
from unfallatlas.models.baseline import (
    build_logreg_pipeline,
    build_majority_class_classifier,
    build_random_guess_classifier,
)


def _toy_X_y(n=80):
    rng = np.random.default_rng(0)
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


def test_majority_class_classifier_always_predicts_the_mode():
    X, y = _toy_X_y()
    clf = build_majority_class_classifier()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(preds) == {y.mode().iloc[0]}


def test_random_guess_classifier_predicts_within_known_classes():
    X, y = _toy_X_y()
    clf = build_random_guess_classifier()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(preds).issubset({1, 2, 3})


def test_logreg_pipeline_fits_and_predicts_known_labels():
    X, y = _toy_X_y()
    preprocessor = build_preprocessor(scale_for_linear=True)
    pipe = build_logreg_pipeline(preprocessor)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({1, 2, 3})
