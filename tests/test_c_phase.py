import pandas as pd

from unfallatlas.models.c_phase import (
    build_inference_contract,
    build_qualitative_matrix,
    compute_error_slices,
    decode_slice_label,
    humanize_feature_name,
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
    feature_frame = pd.DataFrame({"LAT": [50.1, 52.3], "LON": [7.2, 13.4]})
    contract = build_inference_contract(
        feature_columns=["LAT", "LON"],
        dtypes={"LAT": "float64", "LON": "float64"},
        model_card=model_card,
        feature_frame=feature_frame,
    )
    assert contract["threshold"] == 0.4986
    assert contract["required_columns"] == [
        {
            "name": "LAT",
            "dtype": "float64",
            "source": "Unfallatlas raw/engineered (U-phase)",
            "min": 50.1,
            "max": 52.3,
        },
        {
            "name": "LON",
            "dtype": "float64",
            "source": "Unfallatlas raw/engineered (U-phase)",
            "min": 7.2,
            "max": 13.4,
        },
    ]
    assert "generated_at_utc" in contract


def test_build_inference_contract_source_and_categories():
    model_card = {
        "optimal_threshold_val_2023": 0.4986,
        "target_encoding": "1 = KSI (UKATGEORIE in {1,2}), 0 = slight (UKATGEORIE = 3)",
    }
    feature_frame = pd.DataFrame(
        {
            "osm_road_density": [1.0, 2.0, 3.0],
            "dwd_precip_mm": [0.0, 1.0, 2.0],
            "STRZUSTAND": ["0", "1", "2"],
            "IstRad": [True, False, True],
        }
    )
    contract = build_inference_contract(
        feature_columns=["osm_road_density", "dwd_precip_mm", "STRZUSTAND", "IstRad"],
        dtypes={
            "osm_road_density": "float64",
            "dwd_precip_mm": "float64",
            "STRZUSTAND": "object",
            "IstRad": "bool",
        },
        model_card=model_card,
        feature_frame=feature_frame,
    )
    by_name = {c["name"]: c for c in contract["required_columns"]}
    assert by_name["osm_road_density"]["source"] == "OSM road-context enrichment (U-phase)"
    assert by_name["dwd_precip_mm"]["source"] == "DWD weather enrichment (U-phase)"
    assert by_name["STRZUSTAND"]["categories"] == ["0", "1", "2"]
    assert by_name["IstRad"]["categories"] == [True, False]


def test_build_inference_contract_high_cardinality_column_gets_a_note():
    model_card = {
        "optimal_threshold_val_2023": 0.4986,
        "target_encoding": "1 = KSI (UKATGEORIE in {1,2}), 0 = slight (UKATGEORIE = 3)",
    }
    feature_frame = pd.DataFrame({"UKREIS": [str(i) for i in range(60)]})
    contract = build_inference_contract(
        feature_columns=["UKREIS"],
        dtypes={"UKREIS": "object"},
        model_card=model_card,
        feature_frame=feature_frame,
    )
    entry = contract["required_columns"][0]
    assert "categories" not in entry
    assert "60 unique values" in entry["note"]


def test_decode_slice_label_decodes_coded_categorical():
    assert decode_slice_label("UART", 2) == "Unfallart: Zzs. vorausfahrendes Fz. (Auffahrunfall)"


def test_decode_slice_label_handles_string_valued_code():
    # slice values arrive as whatever dtype the source column has (often str
    # after groupby); decoding must not require the caller to pre-cast.
    assert decode_slice_label("STRZUSTAND", "2") == "Straßenzustand: Winterglatt"


def test_decode_slice_label_formats_hour_column():
    assert decode_slice_label("USTUNDE", 7) == "Uhrzeit: 07:00 Uhr"


def test_decode_slice_label_passes_through_uncoded_column():
    assert decode_slice_label("osm_dominant_road_class", "residential") == (
        "Straßenklasse (OSM): residential"
    )


def test_decode_slice_label_falls_back_to_raw_value_for_unknown_code():
    assert decode_slice_label("UART", 999) == "Unfallart: 999"


def test_humanize_feature_name_decodes_onehot_dummy():
    assert humanize_feature_name("UART_2") == (
        "Unfallart: Zzs. vorausfahrendes Fz. (Auffahrunfall)"
    )
    assert humanize_feature_name("UTYP1_1") == "Unfalltyp: Fahrunfall"


def test_humanize_feature_name_passes_through_other_columns():
    assert humanize_feature_name("osm_way_count") == "osm_way_count"
    assert humanize_feature_name("IstKrad") == "IstKrad"
