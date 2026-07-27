import pandas as pd
import pytest

from unfallatlas.viz.streamlit_app import (
    DEFAULT_WIDGET_VALUES,
    FEATURE_DISPLAY_NAMES,
    UART_LABELS,
    build_input_row,
    decode_feature_value,
    display_feature_name,
    get_column_spec,
    load_3class_comparison,
    load_binary_comparison,
    load_candidate_metrics,
    load_categorical_options,
    load_champion_model,
    load_inference_contract,
    load_model_card,
    load_permutation_importance,
    load_severity_grid,
    predict_ksi,
)


def test_load_inference_contract_has_required_keys():
    contract = load_inference_contract()
    assert contract["model_path"] == "data/processed/a3_binary_best_model.joblib"
    assert contract["threshold"] == 0.49860217036273086
    assert len(contract["required_columns"]) == 30
    assert contract["required_columns"][0]["name"] == "UREGBEZ"


def test_load_model_card_has_test_2024_metrics():
    card = load_model_card()
    assert card["optimal_threshold_val_2023"] == 0.49860217036273086
    metrics = card["test_2024_metrics"]
    assert metrics["macro_f1"] == 0.6038956179272812
    assert metrics["confusion_matrix"] == [[22767, 21431], [51887, 172434]]


def test_confusion_matrix_row_order_matches_recall_ksi():
    card = load_model_card()
    cm = card["test_2024_metrics"]["confusion_matrix"]
    recall_ksi_from_cm = cm[0][0] / sum(cm[0])
    assert recall_ksi_from_cm == pytest.approx(card["test_2024_metrics"]["recall_ksi"])


def test_load_binary_comparison_has_ten_candidates():
    df = load_binary_comparison()
    assert len(df) == 10
    assert {"model", "macro_f1", "recall_ksi"}.issubset(df.columns)


def test_load_3class_comparison_has_nineteen_configs():
    df = load_3class_comparison()
    assert len(df) == 19
    assert {"model", "macro_f1", "recall_class_1"}.issubset(df.columns)


def test_load_candidate_metrics_parses_confusion_matrix_as_list():
    df = load_candidate_metrics()
    assert len(df) == 10
    first_cm = df.loc[df["model"] == "binary_random_guess", "confusion_matrix"].iloc[0]
    assert first_cm == [[22976, 23053], [111714, 111305]]


def test_load_permutation_importance_returns_top_15_sorted_by_rank():
    df = load_permutation_importance()
    assert len(df) == 15
    assert (df["model"] == "binary_random_forest_balanced").all()
    assert df["rank"].is_monotonic_increasing


def test_load_permutation_importance_respects_top_n():
    df = load_permutation_importance(top_n=3)
    assert len(df) == 3


def test_load_categorical_options_ukreis_has_87_sorted_values():
    options = load_categorical_options("UKREIS")
    assert len(options) == 87
    assert options[0] == "01"
    assert options == sorted(options)


def test_get_column_spec_returns_matching_entry():
    contract = load_inference_contract()
    spec = get_column_spec(contract, "UMONAT")
    assert spec["name"] == "UMONAT"
    assert spec["min"] == 1.0
    assert spec["max"] == 12.0


def test_get_column_spec_raises_on_unknown_column():
    contract = load_inference_contract()
    try:
        get_column_spec(contract, "NOT_A_COLUMN")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_default_widget_values_cover_every_required_column():
    contract = load_inference_contract()
    required_names = {col["name"] for col in contract["required_columns"]}
    assert required_names.issubset(DEFAULT_WIDGET_VALUES.keys())


class _FakePipeline:
    """Tiny stand-in for the real 407 MB joblib pipeline, for fast unit tests."""

    def predict_proba(self, row):
        return [[0.3, 0.7]]


def test_build_input_row_keeps_istgkfz_as_real_bool_like_its_siblings():
    contract = load_inference_contract()
    row = build_input_row(DEFAULT_WIDGET_VALUES, contract)
    assert row.loc[0, "IstGkfz"] is False


def test_build_input_row_keeps_real_bools_for_other_ist_columns():
    contract = load_inference_contract()
    row = build_input_row(DEFAULT_WIDGET_VALUES, contract)
    assert row.loc[0, "IstPKW"] is True
    assert row.loc[0, "IstRad"] is False


def test_build_input_row_has_all_30_required_columns_in_contract_order():
    contract = load_inference_contract()
    row = build_input_row(DEFAULT_WIDGET_VALUES, contract)
    expected_order = [col["name"] for col in contract["required_columns"]]
    assert list(row.columns) == expected_order


def test_build_input_row_raises_clear_keyerror_on_missing_widget_value():
    contract = load_inference_contract()
    incomplete_values = {k: v for k, v in DEFAULT_WIDGET_VALUES.items() if k != "UMONAT"}
    try:
        build_input_row(incomplete_values, contract)
        assert False, "expected KeyError"
    except KeyError as exc:
        assert "UMONAT" in str(exc)


def test_predict_ksi_applies_threshold_not_default_half():
    row = build_input_row(DEFAULT_WIDGET_VALUES, load_inference_contract())
    proba, prediction = predict_ksi(_FakePipeline(), row, threshold=0.75)
    assert proba == 0.7
    assert prediction == 0  # 0.7 < 0.75 threshold, even though 0.7 > sklearn's default 0.5

    proba2, prediction2 = predict_ksi(_FakePipeline(), row, threshold=0.5)
    assert prediction2 == 1  # 0.7 >= 0.5


def test_load_champion_model_predicts_on_real_contract_row():
    """End-to-end check that the committed joblib model, the committed
    inference contract, and build_input_row/predict_ksi all agree - this is
    the concrete proof that the app needs no notebook execution."""
    contract = load_inference_contract()
    model = load_champion_model()
    row = build_input_row(DEFAULT_WIDGET_VALUES, contract)
    proba, prediction = predict_ksi(model, row, contract["threshold"])
    assert 0.0 <= proba <= 1.0
    assert prediction in (0, 1)


def test_display_feature_name_maps_known_column():
    assert display_feature_name("dwd_wind_speed_ms") == "Wind Speed (m/s)"


def test_display_feature_name_falls_back_to_raw_name_for_unknown_column():
    assert display_feature_name("some_future_column") == "some_future_column"


def test_feature_display_names_covers_every_permutation_importance_feature():
    importance_df = pd.read_csv("data/processed/c_phase_permutation_importance.csv")
    all_features = set(importance_df["feature"].unique())
    assert all_features.issubset(FEATURE_DISPLAY_NAMES.keys())


def test_decode_feature_value_maps_uart_code_to_label():
    assert decode_feature_value("UART", 6) == UART_LABELS[6]


def test_decode_feature_value_passes_through_unmapped_column():
    assert decode_feature_value("dwd_wind_speed_ms", 3.0) == 3.0


def test_load_severity_grid_has_expected_columns_and_counts_agree():
    df = load_severity_grid()
    assert {"lat_bin", "lon_bin", "ksi_count", "slight_count", "total"}.issubset(df.columns)
    assert len(df) > 0
    assert (df["ksi_count"] + df["slight_count"] == df["total"]).all()


def test_load_severity_grid_respects_precision_parameter():
    coarse = load_severity_grid(precision=0.5)
    fine = load_severity_grid(precision=0.1)
    assert len(coarse) < len(fine)


def test_build_severity_map_returns_a_folium_map():
    import folium

    from unfallatlas.viz.streamlit_app import build_severity_map

    m = build_severity_map()
    assert isinstance(m, folium.Map)
