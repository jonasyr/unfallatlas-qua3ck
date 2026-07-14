import pickle

import numpy as np
import pandas as pd
import pytest

from unfallatlas.features.preprocessing import (
    TargetMeanEncoder,
    build_preprocessor,
    chronological_split,
    split_features_target,
    split_features_target_binary,
)


def _toy_frame(n=60):
    rng = np.random.default_rng(42)
    years = np.repeat([2016, 2017, 2022, 2023, 2024], n // 5)
    road_classes = np.array(["primary", "secondary", "residential", "motorway", "tertiary", None])
    return pd.DataFrame(
        {
            "UJAHR": years,
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
            "dwd_station_id": rng.choice(["A123", "B456", "C789"], n),
            "h3_cell": rng.choice([f"cell_{i}" for i in range(10)], n),
            "osm_dominant_road_class": rng.choice(road_classes, n),
            "osm_maxspeed_mean": rng.choice(
                [30.0, 50.0, 70.0, 100.0, np.nan], n, p=[0.2, 0.3, 0.2, 0.2, 0.1]
            ),
            "osm_maxspeed_max": rng.choice(
                [50.0, 70.0, 100.0, 130.0, np.nan], n, p=[0.2, 0.3, 0.2, 0.2, 0.1]
            ),
            "osm_road_density": rng.choice([*rng.exponential(500, n // 2), np.nan], n),
            "osm_way_count": rng.choice([*rng.integers(2, 400, n // 2), np.nan], n),
            "UKATGEORIE": rng.choice([1, 2, 3], n, p=[0.05, 0.25, 0.70]),
        }
    )


def test_chronological_split_respects_year_boundaries():
    df = _toy_frame()
    train, val, test = chronological_split(df)
    assert train["UJAHR"].max() <= 2022
    assert set(val["UJAHR"].unique()) == {2023}
    assert set(test["UJAHR"].unique()) == {2024}
    assert len(train) + len(val) + len(test) == len(df)


def test_split_features_target_drops_year_and_target():
    df = _toy_frame()
    X, y = split_features_target(df)
    assert "UJAHR" not in X.columns
    assert "UKATGEORIE" not in X.columns
    assert y.name == "UKATGEORIE"
    assert set(y.unique()).issubset({1, 2, 3})


def test_target_mean_encoder_smooths_toward_global_mean():
    X = pd.DataFrame({"UKREIS": [1, 1, 1, 2]})
    y = pd.Series([3, 3, 3, 1])  # UKREIS=1 always severity 3, UKREIS=2 only one obs of 1
    enc = TargetMeanEncoder(columns=["UKREIS"], smoothing=10.0)
    enc.fit(X, y)
    out = enc.transform(X)
    # UKREIS=2 has a single observation; smoothing must pull it toward the
    # global mean (2.5) rather than reporting the raw value (1.0).
    assert out["UKREIS_target_enc"].iloc[3] > 1.0
    assert out["UKREIS_target_enc"].iloc[3] < 2.5


def test_target_mean_encoder_unseen_category_gets_global_mean():
    X_train = pd.DataFrame({"UKREIS": [1, 1, 2, 2]})
    y_train = pd.Series([1, 1, 3, 3])
    enc = TargetMeanEncoder(columns=["UKREIS"], smoothing=5.0)
    enc.fit(X_train, y_train)
    out = enc.transform(pd.DataFrame({"UKREIS": [999]}))
    assert out["UKREIS_target_enc"].iloc[0] == pytest.approx(enc.global_mean_)


def test_build_preprocessor_fit_transform_has_no_nans():
    df = _toy_frame()
    train, _, _ = chronological_split(df)
    X, y = split_features_target(train)
    preprocessor = build_preprocessor(scale_for_linear=False)
    transformed = preprocessor.fit_transform(X, y)
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == len(X)


def test_build_preprocessor_scale_for_linear_changes_passthrough_columns():
    df = _toy_frame()
    train, _, _ = chronological_split(df)
    X, y = split_features_target(train)
    tree_pre = build_preprocessor(scale_for_linear=False).fit(X, y)
    linear_pre = build_preprocessor(scale_for_linear=True).fit(X, y)
    tree_out = tree_pre.transform(X)
    linear_out = linear_pre.transform(X)
    # Same number of rows either way; scaling changes values, not row count.
    assert tree_out.shape[0] == linear_out.shape[0] == len(X)


def test_fitted_preprocessor_is_picklable():
    """Regression test: a lambda inside build_preprocessor's dwd_station_dist_km
    log-transform previously made the fitted ColumnTransformer unpicklable
    (joblib.dump raised PicklingError), which only surfaced once the A3
    notebook tried to checkpoint a fitted pipeline to disk. Every model
    artefact this project saves goes through joblib.dump, so this must hold.
    """
    df = _toy_frame()
    train, _, _ = chronological_split(df)
    X, y = split_features_target(train)
    preprocessor = build_preprocessor().fit(X, y)

    pickled = pickle.dumps(preprocessor)
    restored = pickle.loads(pickled)
    restored_out = restored.transform(X)
    original_out = preprocessor.transform(X)

    np.testing.assert_array_equal(restored_out, original_out)


def test_build_preprocessor_encodes_osm_road_class_as_onehot():
    """osm_dominant_road_class (U-phase §10: one-hot, dedicated 'unknown'
    category for missing) must appear as one-hot dummy columns in the
    fitted transformer's output feature names, not be silently dropped."""
    df = _toy_frame()
    train, _, _ = chronological_split(df)
    X, y = split_features_target(train)
    preprocessor = build_preprocessor(scale_for_linear=False).fit(X, y)
    feature_names = preprocessor.get_feature_names_out()
    assert any(name.startswith("osm_dominant_road_class_") for name in feature_names)


def test_build_preprocessor_handles_missing_osm_road_class():
    """_toy_frame() seeds real None values in osm_dominant_road_class
    (mirrors the 0.0065% missing rate observed in the real cached data) -
    fit_transform must not raise and must produce no NaN output."""
    df = _toy_frame()
    train, _, _ = chronological_split(df)
    X, y = split_features_target(train)
    preprocessor = build_preprocessor(scale_for_linear=False)
    transformed = preprocessor.fit_transform(X, y)
    assert not np.isnan(transformed).any()


def test_build_preprocessor_handles_missing_osm_numeric_columns():
    """osm_maxspeed_mean/_max (median-impute) and osm_road_density/
    osm_way_count (zero-fill) must all resolve to finite values even
    though _toy_frame() seeds NaN in all four."""
    df = _toy_frame()
    train, _, _ = chronological_split(df)
    X, y = split_features_target(train)
    assert X["osm_maxspeed_mean"].isna().any()
    assert X["osm_road_density"].isna().any()
    preprocessor = build_preprocessor(scale_for_linear=False)
    transformed = preprocessor.fit_transform(X, y)
    assert not np.isnan(transformed).any()


def test_split_features_target_binary_produces_binary_labels():
    df = _toy_frame(n=60)
    # Add UKATGEORIE column to toy frame
    rng = np.random.default_rng(99)
    df["UKATGEORIE"] = rng.choice([1, 2, 3], len(df), p=[0.01, 0.15, 0.84])
    X, y = split_features_target_binary(df)
    assert set(y.unique()) <= {0, 1}
    assert y[df["UKATGEORIE"].isin([1, 2])].eq(1).all()
    assert y[df["UKATGEORIE"].eq(3)].eq(0).all()
    assert "UKATGEORIE" not in X.columns
    assert "UJAHR" not in X.columns


def test_build_preprocessor_drops_h3_cell_and_dwd_station_id():
    """h3_cell (join key, ~220k distinct values in the real data) and
    dwd_station_id are not features - both must stay silently dropped via
    remainder="drop", exactly like before this plan's OSM wiring. This
    locks in the non-use decision so a future column-list edit can't
    accidentally sweep a high-cardinality ID column into the model."""
    df = _toy_frame()
    train, _, _ = chronological_split(df)
    X, y = split_features_target(train)
    preprocessor = build_preprocessor(scale_for_linear=False).fit(X, y)
    feature_names = list(preprocessor.get_feature_names_out())
    assert not any("h3_cell" in name for name in feature_names)
    assert not any("dwd_station_id" in name for name in feature_names)
