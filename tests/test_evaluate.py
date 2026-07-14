import numpy as np
import pandas as pd

from unfallatlas.models.evaluate import (
    BINARY_MACRO_F1_THRESHOLD,
    BINARY_RECALL_KSI_THRESHOLD,
    MACRO_F1_THRESHOLD,
    RECALL_CLASS_1_THRESHOLD,
    evaluate_binary_predictions,
    evaluate_predictions,
    find_best_binary_threshold,
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


def test_find_best_binary_threshold_recovers_perfect_separator():
    # Scores perfectly separate the two classes at score=0: negatives < 0, positives > 0.
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    threshold, metrics = find_best_binary_threshold(y_true, scores)
    assert metrics["macro_f1"] == 1.0
    assert metrics["recall_ksi"] == 1.0
    # Any threshold strictly between -0.5 and 0.5 reproduces the perfect split.
    assert -0.5 < threshold <= 0.5


def test_find_best_binary_threshold_falls_back_when_gate_unreachable():
    # No positives exist in y_true at all, so recall_ksi is 0.0 at every threshold
    # in the sweep (sklearn's zero_division default for an absent class) - no
    # threshold can ever clear a positive recall_gate, forcing the unconstrained
    # macro-F1 fallback.
    #
    # NOTE: this scenario replaces the brief's original single-positive example
    # (y_true=[0,0,0,0,1], scores=[-1,-0.5,0,0.5,-2], recall_gate=1.0). That
    # example is unreachable-by-design in the *opposite* sense: with >=1 true
    # positive, the sweep always includes threshold=scores.min(), which
    # predicts every row positive and therefore always yields recall_ksi=1.0 -
    # so recall_gate=1.0 was trivially satisfied there, not infeasible, and the
    # test failed against the brief's own reference implementation. Verified via
    # TDD (see task-1-report.md) before substituting this all-negative-y_true
    # scenario, which is the only way to make the gate genuinely unreachable.
    y_true = np.array([0, 0, 0, 0, 0])
    scores = np.array([-1.0, -0.5, 0.0, 0.5, -2.0])
    threshold, metrics = find_best_binary_threshold(y_true, scores, recall_gate=1.0)
    assert metrics["recall_ksi"] < 1.0  # gate was infeasible, unconstrained fallback used
    assert isinstance(threshold, float)


def test_find_best_binary_threshold_works_with_decision_function_range():
    # decision_function output is unbounded (not in [0, 1]) - confirm the sweep
    # range is derived from the scores themselves, not hardcoded to a [0, 1] tube.
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([-10.0, -8.0, 8.0, 10.0])
    threshold, metrics = find_best_binary_threshold(y_true, scores)
    assert metrics["macro_f1"] == 1.0
    assert -8.0 < threshold <= 8.0
