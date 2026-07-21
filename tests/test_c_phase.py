import pandas as pd

from unfallatlas.models.c_phase import (
    build_inference_contract,
    build_qualitative_matrix,
    compute_error_slices,
)


def test_compute_error_slices_basic():
    # index:        0      1      2     3     4     5
    y_true = pd.Series([1, 1, 0, 0, 1, 0])
    y_pred = pd.Series([1, 0, 0, 1, 0, 0])
    slice_frame = pd.DataFrame({"weather": ["rain", "rain", "dry", "dry", "dry", "rain"]})
    # rain = {0,1,5}: idx0 TP, idx1 FN, idx5 TN -> rain n_false_negative=1
    # dry  = {2,3,4}: idx2 TN, idx3 FP, idx4 FN -> dry n_false_positive=1, n_false_negative=1

    result = compute_error_slices(y_true, y_pred, slice_frame, ["weather"])

    assert set(result["slice_column"]) == {"weather"}
    rain_row = result[result["slice_value"] == "rain"].iloc[0]
    dry_row = result[result["slice_value"] == "dry"].iloc[0]
    assert rain_row["n"] == 3
    assert rain_row["n_false_negative"] == 1
    assert dry_row["n"] == 3
    assert dry_row["n_false_positive"] == 1


def test_compute_error_slices_multiple_columns():
    y_true = pd.Series([1, 0])
    y_pred = pd.Series([1, 0])
    slice_frame = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})

    result = compute_error_slices(y_true, y_pred, slice_frame, ["a", "b"])

    assert set(result["slice_column"]) == {"a", "b"}
    assert len(result) == 4  # 2 values each for a and b


def test_build_qualitative_matrix_orders_best_first():
    rows = [
        {
            "model": "random_forest",
            "macro_f1": 0.6026,
            "recall_ksi": 0.5255,
            "latency_ms_per_1k": 50.0,
            "interpretability_score": 0.7,
            "robustness_score": 0.9,
            "training_cost_score": 0.5,
        },
        {
            "model": "xgboost",
            "macro_f1": 0.5699,
            "recall_ksi": 0.6824,
            "latency_ms_per_1k": 20.0,
            "interpretability_score": 0.5,
            "robustness_score": 0.8,
            "training_cost_score": 0.3,
        },
    ]
    result = build_qualitative_matrix(rows)
    assert list(result.columns[:1]) == ["model"]
    assert "weighted_score" in result.columns
    assert result.iloc[0]["weighted_score"] >= result.iloc[1]["weighted_score"]


def test_build_inference_contract_shape():
    model_card = {
        "optimal_threshold_val_2023": 0.4986,
        "target_encoding": "1 = KSI (UKATGEORIE in {1,2}), 0 = slight (UKATGEORIE = 3)",
    }
    contract = build_inference_contract(
        feature_columns=["LAT", "LON"],
        dtypes={"LAT": "float64", "LON": "float64"},
        model_card=model_card,
    )
    assert contract["threshold"] == 0.4986
    assert contract["required_columns"] == [
        {"name": "LAT", "dtype": "float64"},
        {"name": "LON", "dtype": "float64"},
    ]
    assert "generated_at_utc" in contract
