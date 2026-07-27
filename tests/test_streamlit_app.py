from unfallatlas.viz.streamlit_app import (
    load_3class_comparison,
    load_binary_comparison,
    load_candidate_metrics,
    load_inference_contract,
    load_model_card,
    load_permutation_importance,
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
