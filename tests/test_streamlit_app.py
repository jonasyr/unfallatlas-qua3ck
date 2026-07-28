import pandas as pd
import pytest

from unfallatlas.viz.streamlit_app import (
    DEFAULT_WIDGET_VALUES,
    FEATURE_DISPLAY_NAMES,
    RISK_BANDS,
    SEVERITY_COLORS,
    UART_LABELS,
    build_input_row,
    build_severity_base_map,
    build_severity_feature_groups,
    confidence_opacity,
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
    load_national_ksi_rate,
    load_permutation_importance,
    load_severity_grid,
    predict_ksi,
    risk_band_index,
    severity_legend_markdown,
    shrunk_relative_risk,
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


def test_build_severity_base_map_has_no_layers_attached():
    # Layers must reach the browser only via st_folium(feature_group_to_add=...).
    # Anything attached here raises a JS ReferenceError and blanks the whole map.
    # Cache cleared first: st.cache_resource is a process-global singleton, and if
    # this suite runs in the same process as an AppTest render of the Overview page
    # (it does - see tests/test_overview_page.py), st_folium's own internals have
    # already attached layers to the cached instance by the time this test runs.
    # That attachment is legitimate (st_folium wires it up correctly, unlike the
    # banned manual .add_to() pattern this test guards against) - clearing the
    # cache here just ensures this assertion inspects a freshly built map.
    import folium

    build_severity_base_map.clear()
    base_map = build_severity_base_map()
    children = base_map._children.values()
    assert not any(isinstance(child, folium.FeatureGroup) for child in children)
    assert not any(isinstance(child, folium.LayerControl) for child in children)


def test_build_severity_feature_groups_returns_one_named_group_per_band():
    groups = build_severity_feature_groups()
    assert len(groups) == len(RISK_BANDS)
    assert [group.layer_name for group in groups] == [band.label for band in RISK_BANDS]


def test_build_severity_feature_groups_covers_every_grid_cell_exactly_once():
    # Cache cleared first: st_folium's internal render() adds one bookkeeping
    # child per group the first time it renders a group (idempotent on repeat
    # renders, but it would otherwise inflate this count by 1 per group if an
    # AppTest render of the Overview page already ran in this process - see
    # tests/test_overview_page.py - since st.cache_resource is a process-global
    # singleton shared between AppTest and this direct call).
    build_severity_feature_groups.clear()
    groups = build_severity_feature_groups()
    circles_per_group = [len(group._children) for group in groups]
    assert sum(circles_per_group) == 4857
    assert circles_per_group == [239, 1137, 2014, 1263, 204]


def test_build_severity_map_is_gone():
    # Replaced by build_severity_base_map + build_severity_feature_groups.
    import unfallatlas.viz.streamlit_app as module

    assert not hasattr(module, "build_severity_map")


def test_shrunk_relative_risk_returns_baseline_ratio_for_empty_cell():
    # A cell with no accidents is pure prior: shrunk rate == baseline, ratio == 1.0
    assert shrunk_relative_risk(0, 0, 0.1891444326398238) == pytest.approx(1.0)


def test_shrunk_relative_risk_converges_to_raw_rate_for_large_cell():
    # 100_000 accidents at a 40% KSI rate: shrinkage of k=20 is negligible
    result = shrunk_relative_risk(40_000, 100_000, 0.1891444326398238)
    raw_ratio = 0.4 / 0.1891444326398238
    assert result == pytest.approx(raw_ratio, rel=1e-3)


def test_shrunk_relative_risk_pulls_small_noisy_cell_toward_baseline():
    # 1 accident, and it was KSI. Raw rate is 100% (5.29x baseline) - pure noise.
    # Shrinkage must pull it far down, below the "very high" 2.0x band threshold.
    baseline = 0.1891444326398238
    raw_ratio = 1.0 / baseline
    shrunk = shrunk_relative_risk(1, 1, baseline)
    assert raw_ratio > 5.0
    assert shrunk < 2.0


def test_shrunk_relative_risk_uses_explicit_k():
    baseline = 0.2
    # (ksi + k*baseline) / (total + k) / baseline
    # k=0 disables shrinkage entirely -> raw ratio
    assert shrunk_relative_risk(5, 10, baseline, k=0) == pytest.approx(0.5 / 0.2)


def test_shrunk_relative_risk_rejects_non_positive_baseline():
    with pytest.raises(ValueError, match="baseline"):
        shrunk_relative_risk(1, 10, 0.0)


def test_risk_bands_are_five_contiguous_ascending_ranges():
    assert len(RISK_BANDS) == 5
    assert RISK_BANDS[0].lower == 0.0
    assert RISK_BANDS[-1].upper == float("inf")
    for lower_band, upper_band in zip(RISK_BANDS[:-1], RISK_BANDS[1:], strict=True):
        # contiguous: no gaps, no overlaps
        assert lower_band.upper == upper_band.lower


def test_risk_bands_endpoints_reuse_the_app_severity_palette():
    assert RISK_BANDS[0].color == SEVERITY_COLORS["slight"]
    assert RISK_BANDS[-1].color == SEVERITY_COLORS["KSI"]


def test_risk_bands_all_have_distinct_colors_and_labels():
    assert len({band.color for band in RISK_BANDS}) == 5
    assert len({band.label for band in RISK_BANDS}) == 5


@pytest.mark.parametrize(
    ("relative_risk", "expected_index"),
    [
        (0.0, 0),
        (0.74, 0),
        (0.75, 1),  # boundaries are half-open [lower, upper): 0.75 lands in band 1
        (1.0, 1),
        (1.09, 1),
        (1.1, 2),
        (1.49, 2),
        (1.5, 3),
        (1.99, 3),
        (2.0, 4),
        (100.0, 4),
    ],
)
def test_risk_band_index_uses_half_open_intervals(relative_risk, expected_index):
    assert risk_band_index(relative_risk) == expected_index


def test_risk_band_index_rejects_negative_risk():
    with pytest.raises(ValueError, match="relative_risk"):
        risk_band_index(-0.1)


@pytest.mark.parametrize(
    ("total", "expected_opacity"),
    [(0, 0.25), (19, 0.25), (20, 0.45), (99, 0.45), (100, 0.65), (25_916, 0.65)],
)
def test_confidence_opacity_tiers(total, expected_opacity):
    assert confidence_opacity(total) == expected_opacity


def test_severity_legend_markdown_states_every_band_and_the_baseline():
    legend = severity_legend_markdown()
    for band in RISK_BANDS:
        assert band.label in legend
        assert band.color in legend
    # The national baseline must be stated as a percentage, so "2x average" is readable
    assert "18.9" in legend
    # The opacity convention must be explained, not left implicit
    assert "20" in legend and "100" in legend


def test_load_national_ksi_rate_matches_the_measured_value():
    # 395766 KSI rows out of 2092401 in the committed parquet
    assert load_national_ksi_rate() == pytest.approx(0.1891444326398238)


def test_load_severity_grid_has_expected_shape_and_columns():
    grid = load_severity_grid()
    assert len(grid) == 4857
    assert {
        "lat_bin",
        "lon_bin",
        "center_lat",
        "center_lon",
        "ksi_count",
        "slight_count",
        "total",
    }.issubset(grid.columns)


def test_load_severity_grid_counts_sum_to_the_full_dataset():
    grid = load_severity_grid()
    assert grid["total"].sum() == 2092401
    assert (grid["ksi_count"] + grid["slight_count"] == grid["total"]).all()


def test_load_severity_grid_centroids_lie_inside_their_own_cell():
    # A centroid of points that all round to the same 0.1-degree bin cannot be more
    # than half a bin width away from that bin's coordinate. Verified: 0 violations.
    grid = load_severity_grid()
    assert ((grid["center_lat"] - grid["lat_bin"]).abs() <= 0.05).all()
    assert ((grid["center_lon"] - grid["lon_bin"]).abs() <= 0.05).all()


def test_load_severity_grid_centroids_actually_differ_from_bin_coordinates():
    # Guards against a regression that silently aliases center_lat back to lat_bin.
    grid = load_severity_grid()
    assert (grid["center_lat"] != grid["lat_bin"]).sum() > 4000


def test_severity_grid_bands_populate_every_band():
    # The whole point of the redesign: all five bands carry cells, instead of the
    # old 50%-majority rule that colored only 123 of 4857 cells red.
    grid = load_severity_grid()
    baseline = load_national_ksi_rate()
    indices = [
        risk_band_index(shrunk_relative_risk(row.ksi_count, row.total, baseline))
        for row in grid.itertuples()
    ]
    counts = {index: indices.count(index) for index in range(5)}
    assert counts == {0: 239, 1: 1137, 2: 2014, 3: 1263, 4: 204}
