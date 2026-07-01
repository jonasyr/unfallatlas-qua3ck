"""Ordinal classification via Frank & Hall (2001) rank decomposition.

U-phase §10: "A³ chooses the mitigation ... ordinal classification" — the
target has a natural order (Q-phase §5), so this is one of the four
imbalance/ordering strategies compared on the champion model, not a
general-purpose model family on its own.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline


class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """Decomposes a K-class ordinal problem into K-1 binary classifiers.

    Binary classifier i predicts P(y > classes_[i]). Class probabilities
    are recovered by differencing consecutive cumulative probabilities.
    ``base_estimator`` must expose ``predict_proba``.
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        if len(self.classes_) < 3:
            raise ValueError("OrdinalClassifier requires at least 3 ordered classes.")
        self.binary_estimators_ = []
        for threshold in self.classes_[:-1]:
            binary_target = (np.asarray(y) > threshold).astype(int)
            estimator = clone(self.base_estimator)
            estimator.fit(X, binary_target)
            self.binary_estimators_.append(estimator)
        return self

    def predict_proba(self, X) -> np.ndarray:
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        # cum_probs[:, i] = P(y > classes_[i-1]), decreasing from 1 (i=0, "P(y > -inf)")
        # down to 0 (i=K, "P(y > classes_[-1])"). Class i's probability is the drop
        # between consecutive cumulative terms: P(y == classes_[i]) = cum_probs[i] - cum_probs[i+1].
        cum_probs = np.column_stack(
            [np.ones(n)]
            + [est.predict_proba(X)[:, 1] for est in self.binary_estimators_]
            + [np.zeros(n)]
        )
        class_probs = np.clip(-np.diff(cum_probs, axis=1), 0, None)
        row_sums = class_probs.sum(axis=1, keepdims=True)
        return class_probs / row_sums

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def build_ordinal_pipeline(preprocessor, base_estimator) -> Pipeline:
    """Wrap ``preprocessor`` + ``OrdinalClassifier(base_estimator)`` in one Pipeline."""
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classify", OrdinalClassifier(base_estimator=base_estimator)),
        ]
    )
