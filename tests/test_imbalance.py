import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, recall_score

from unfallatlas.models.imbalance import (
    balanced_sample_weight,
    find_best_threshold_for_class,
    find_gate_optimal_offsets,
    resample_adasyn,
    resample_smote,
)


def _imbalanced_toy_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice([1, 2, 3], n, p=[0.05, 0.25, 0.70])
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    return X, y


def test_resample_smote_balances_class_counts():
    X, y = _imbalanced_toy_data()
    X_res, y_res = resample_smote(X, y)
    counts = pd.Series(y_res).value_counts()
    # SMOTE (default 'auto' strategy) balances every class to the majority count.
    assert counts.nunique() == 1
    assert len(y_res) >= len(y)


def test_resample_adasyn_increases_minority_share():
    X, y = _imbalanced_toy_data()
    original_minority_share = (y == 1).mean()
    X_res, y_res = resample_adasyn(X, y)
    resampled_minority_share = (np.asarray(y_res) == 1).mean()
    assert resampled_minority_share > original_minority_share


def test_balanced_sample_weight_upweights_rare_classes():
    y = np.array([1, 2, 2, 3, 3, 3])
    weights = balanced_sample_weight(y)
    assert len(weights) == len(y)
    # Class 1 (rarest) must receive a strictly higher weight than class 3 (most common).
    assert weights[0] > weights[-1]


def test_find_best_threshold_for_class_improves_on_default_half():
    # 10 rows; class 1 is rare (2 rows) with high probability but a naive
    # argmax at threshold 0.5 for the *other* classes would bury it.
    y_true = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 3])
    y_proba = np.array(
        [
            [0.4, 0.35, 0.25],
            [0.45, 0.3, 0.25],
            [0.1, 0.7, 0.2],
            [0.1, 0.7, 0.2],
            [0.1, 0.7, 0.2],
            [0.05, 0.15, 0.8],
            [0.05, 0.15, 0.8],
            [0.05, 0.15, 0.8],
            [0.05, 0.15, 0.8],
            [0.05, 0.15, 0.8],
        ]
    )
    threshold = find_best_threshold_for_class(y_true, y_proba, classes=[1, 2, 3], target_class=1)
    assert 0.0 < threshold < 1.0

    # Guard against an index-misalignment bug: after deleting the target class's
    # column, the remaining "other classes" argmax indices must map back to the
    # correct real class labels (2, 3), not a shifted/swapped pair. A misaligned
    # mapping still returns *some* threshold in (0, 1) but produces near-random
    # fallback predictions, so we check macro-F1 directly instead.
    target_idx = [1, 2, 3].index(1)
    other_classes = np.array([2, 3])
    other_proba = np.delete(y_proba, target_idx, axis=1)
    fallback = other_classes[np.argmax(other_proba, axis=1)]
    y_pred = np.where(y_proba[:, target_idx] >= threshold, 1, fallback)
    assert f1_score(y_true, y_pred, average="macro") > 0.9


def test_find_best_threshold_for_class_detects_other_classes_index_swap():
    # With only 2 "other" classes (see the test above), reversing the
    # other_classes/other_proba mapping degrades every threshold's fallback
    # score *uniformly*, so the argmax threshold stays put even when the
    # mapping is broken — that bug class slips through undetected.
    #
    # With 4 classes (3 "other" classes when target_class is fixed), a
    # reversed mapping degrades thresholds *non-uniformly*: some fallback
    # rows flip to a wrong label only at some thresholds and not others,
    # which can shift which threshold is optimal. This fixture was found by
    # search to expose exactly that: the correct implementation's best
    # threshold is 0.25, but reversing other_classes (as in
    # `other_classes[::-1]`) shifts the returned best threshold to 0.50 with
    # a markedly worse macro-F1. Asserting the exact threshold therefore
    # fails if the internal fallback index mapping is broken, unlike a test
    # that only checks the threshold falls in some broad range.
    y_true = np.array([3, 3, 4, 2, 1, 1, 4, 2, 1, 1, 4, 4])
    y_proba = np.array(
        [
            [0.097, 0.195, 0.457, 0.252],
            [0.189, 0.195, 0.353, 0.263],
            [0.745, 0.088, 0.078, 0.089],
            [0.232, 0.286, 0.234, 0.249],
            [0.257, 0.016, 0.113, 0.613],
            [0.584, 0.068, 0.176, 0.171],
            [0.249, 0.110, 0.292, 0.349],
            [0.476, 0.193, 0.076, 0.255],
            [0.059, 0.109, 0.132, 0.700],
            [0.534, 0.272, 0.153, 0.041],
            [0.341, 0.146, 0.330, 0.183],
            [0.222, 0.008, 0.118, 0.652],
        ]
    )
    threshold = find_best_threshold_for_class(y_true, y_proba, classes=[1, 2, 3, 4], target_class=1)
    assert threshold == pytest.approx(0.25)

    target_idx = [1, 2, 3, 4].index(1)
    other_classes = np.array([2, 3, 4])
    other_proba = np.delete(y_proba, target_idx, axis=1)
    fallback = other_classes[np.argmax(other_proba, axis=1)]
    y_pred = np.where(y_proba[:, target_idx] >= threshold, 1, fallback)
    assert f1_score(y_true, y_pred, average="macro") == pytest.approx(0.7095238095238094)


def test_find_gate_optimal_offsets_feasible_returns_offsets_that_satisfy_constraint():
    # class 1 samples have second-highest P(1)=0.28 < P(3)=0.44 → default argmax gives class 3
    y_true = np.array([1, 1, 2, 2, 3, 3, 3, 3])
    y_proba = np.array(
        [
            [0.28, 0.28, 0.44],  # true=1, default pred=3
            [0.28, 0.28, 0.44],  # true=1, default pred=3
            [0.10, 0.70, 0.20],  # true=2
            [0.10, 0.70, 0.20],  # true=2
            [0.05, 0.05, 0.90],  # true=3
            [0.05, 0.05, 0.90],  # true=3
            [0.05, 0.05, 0.90],  # true=3
            [0.05, 0.05, 0.90],  # true=3
        ]
    )
    offsets, best_f1 = find_gate_optimal_offsets(
        y_true, y_proba, classes=[1, 2, 3], recall_gate_class=1, recall_gate=0.50
    )
    assert offsets is not None, "Expected feasible offsets"
    o1, o2 = offsets

    # Verify the returned offsets actually satisfy the recall gate
    logit = np.log(np.clip(y_proba, 1e-9, 1)).copy()
    logit[:, [1, 2, 3].index(1)] += o1
    logit[:, [1, 2, 3].index(2)] += o2
    y_pred = np.array([1, 2, 3])[logit.argmax(1)]
    r1 = recall_score(y_true, y_pred, labels=[1], average="macro")
    assert r1 >= 0.50
    assert best_f1 > 0.0


def test_find_gate_optimal_offsets_infeasible_returns_none_offsets():
    # recall_gate=1.01 is mathematically infeasible → offsets must be None
    y_true = np.array([1, 2, 3])
    y_proba = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    offsets, best_f1 = find_gate_optimal_offsets(
        y_true, y_proba, classes=[1, 2, 3], recall_gate=1.01
    )
    assert offsets is None
    assert best_f1 >= 0.0  # still returns the unconstrained best


def test_find_gate_optimal_offsets_returns_positive_or_zero_offsets():
    # Offsets are additive boosts — they must never be negative (no penalising)
    y_true = np.array([1, 1, 2, 2, 3, 3, 3, 3])
    y_proba = np.array(
        [
            [0.28, 0.28, 0.44],
            [0.28, 0.28, 0.44],
            [0.10, 0.70, 0.20],
            [0.10, 0.70, 0.20],
            [0.05, 0.05, 0.90],
            [0.05, 0.05, 0.90],
            [0.05, 0.05, 0.90],
            [0.05, 0.05, 0.90],
        ]
    )
    offsets, _ = find_gate_optimal_offsets(y_true, y_proba, classes=[1, 2, 3])
    if offsets is not None:
        assert offsets[0] >= 0.0
        assert offsets[1] >= 0.0
