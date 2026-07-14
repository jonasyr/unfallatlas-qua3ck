import numpy as np
import pandas as pd

from unfallatlas.models.evaluate import (
    BINARY_MACRO_F1_THRESHOLD,
    BINARY_RECALL_KSI_THRESHOLD,
    MACRO_F1_THRESHOLD,
    RECALL_CLASS_1_THRESHOLD,
    evaluate_binary_predictions,
    evaluate_predictions,
    macro_f1,
    meets_acceptance_criteria,
    meets_binary_acceptance_criteria,
    recall_for_class,
    select_best_candidate,
)


def test_macro_f1_perfect_predictions_is_one():
    y = np.array([1, 2, 3, 1, 2, 3])
    assert macro_f1(y, y) == 1.0


def test_recall_for_class_1_zero_when_never_predicted():
    y_true = np.array([1, 1, 2, 3, 3])
    y_pred = np.array([3, 3, 2, 3, 3])  # class 1 never predicted
    assert recall_for_class(y_true, y_pred, target_class=1) == 0.0


def test_evaluate_predictions_returns_all_expected_keys():
    y = np.array([1, 2, 3, 1, 2, 3])
    metrics = evaluate_predictions(y, y)
    assert set(metrics) == {
        "macro_f1",
        "recall_class_1",
        "recall_class_2",
        "recall_class_3",
        "confusion_matrix",
    }
    assert metrics["macro_f1"] == 1.0
    assert metrics["recall_class_1"] == 1.0


def test_meets_acceptance_criteria_requires_both_thresholds():
    passing = {"macro_f1": MACRO_F1_THRESHOLD, "recall_class_1": RECALL_CLASS_1_THRESHOLD}
    failing_macro = {
        "macro_f1": MACRO_F1_THRESHOLD - 0.01,
        "recall_class_1": RECALL_CLASS_1_THRESHOLD,
    }
    failing_recall = {
        "macro_f1": MACRO_F1_THRESHOLD,
        "recall_class_1": RECALL_CLASS_1_THRESHOLD - 0.01,
    }
    assert meets_acceptance_criteria(passing) is True
    assert meets_acceptance_criteria(failing_macro) is False
    assert meets_acceptance_criteria(failing_recall) is False


def test_meets_acceptance_criteria_majority_class_baseline_fails():
    # Majority-class prediction: everything predicted as class 3.
    y_true = np.array([1] * 10 + [2] * 180 + [3] * 810)
    y_pred = np.array([3] * len(y_true))
    metrics = evaluate_predictions(y_true, y_pred)
    assert meets_acceptance_criteria(metrics) is False


def test_select_best_candidate_picks_highest_macro_f1_among_recall_passers():
    rows = pd.DataFrame(
        [
            {"model": "a", "macro_f1": 0.50, "recall_class_1": 0.60},
            {"model": "b", "macro_f1": 0.60, "recall_class_1": 0.30},  # fails recall gate
            {"model": "c", "macro_f1": 0.45, "recall_class_1": 0.55},
        ]
    )
    winner = select_best_candidate(rows)
    assert winner["model"] == "a"  # highest macro_f1 among recall>=0.5 rows (a, c) is a


def test_select_best_candidate_falls_back_to_combined_score_if_none_pass():
    rows = pd.DataFrame(
        [
            {"model": "a", "macro_f1": 0.50, "recall_class_1": 0.10},  # combined 0.30
            {"model": "b", "macro_f1": 0.40, "recall_class_1": 0.45},  # combined 0.425
        ]
    )
    winner = select_best_candidate(rows)
    assert winner["model"] == "b"


def test_select_best_candidate_custom_threshold():
    rows = pd.DataFrame(
        [
            {"model": "a", "macro_f1": 0.50, "recall_class_1": 0.42},
            {"model": "b", "macro_f1": 0.45, "recall_class_1": 0.20},
        ]
    )
    winner = select_best_candidate(rows, recall_threshold=0.40)
    assert winner["model"] == "a"


def test_evaluate_binary_predictions_returns_all_expected_keys():
    y = np.array([0, 1, 0, 1])
    metrics = evaluate_binary_predictions(y, y)
    assert set(metrics) == {"macro_f1", "recall_ksi", "recall_slight", "confusion_matrix"}
    assert metrics["macro_f1"] == 1.0
    assert metrics["recall_ksi"] == 1.0
    assert metrics["recall_slight"] == 1.0


def test_meets_binary_acceptance_criteria_requires_both_thresholds():
    passing = {"macro_f1": BINARY_MACRO_F1_THRESHOLD, "recall_ksi": BINARY_RECALL_KSI_THRESHOLD}
    failing_f1 = {
        "macro_f1": BINARY_MACRO_F1_THRESHOLD - 0.01,
        "recall_ksi": BINARY_RECALL_KSI_THRESHOLD,
    }
    failing_recall = {
        "macro_f1": BINARY_MACRO_F1_THRESHOLD,
        "recall_ksi": BINARY_RECALL_KSI_THRESHOLD - 0.01,
    }
    assert meets_binary_acceptance_criteria(passing) is True
    assert meets_binary_acceptance_criteria(failing_f1) is False
    assert meets_binary_acceptance_criteria(failing_recall) is False


def test_meets_binary_acceptance_criteria_majority_baseline_fails():
    y_true = np.array([0] * 836 + [1] * 164)
    y_pred = np.array([0] * len(y_true))  # majority-class baseline
    metrics = evaluate_binary_predictions(y_true, y_pred)
    assert meets_binary_acceptance_criteria(metrics) is False
