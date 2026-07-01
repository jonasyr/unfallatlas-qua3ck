"""Stufe 1 tree ensembles — docs/project/PROJEKTPLAN_SETUP.md ML-Roadmap."""

from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


class _ZeroIndexedXGBClassifier(BaseEstimator, ClassifierMixin):
    """Shifts UKATGEORIE's {1,2,3} labels to XGBoost's required {0,1,2} and back.

    XGBClassifier with an explicit ``num_class``/``multi:softprob`` objective
    requires zero-indexed labels and raises ``ValueError: Invalid classes
    inferred from unique values of y`` otherwise (confirmed empirically for
    the pinned xgboost version — labels are not remapped internally). This
    wrapper is the fix noted as a possible requirement in the plan.
    """

    def __init__(self, estimator: XGBClassifier):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        self.estimator_ = clone(self.estimator)
        y_arr = np.asarray(y)
        self.classes_ = np.sort(np.unique(y_arr))
        label_to_zero_indexed = {label: i for i, label in enumerate(self.classes_)}
        y_zero_indexed = np.array([label_to_zero_indexed[label] for label in y_arr])
        if "sample_weight" in fit_params:
            self.estimator_.fit(X, y_zero_indexed, sample_weight=fit_params["sample_weight"])
        else:
            self.estimator_.fit(X, y_zero_indexed)
        return self

    def predict(self, X) -> np.ndarray:
        zero_indexed_preds = self.estimator_.predict(X)
        return self.classes_[zero_indexed_preds]

    def predict_proba(self, X) -> np.ndarray:
        return self.estimator_.predict_proba(X)


def build_random_forest_pipeline(
    preprocessor, class_weight: str | dict | None = "balanced"
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classify",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight=class_weight,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(preprocessor) -> Pipeline:
    """XGBoost has no ``class_weight``; the class-weighted configuration is
    applied via ``sample_weight`` at ``.fit()`` time in the notebook
    (computed by ``unfallatlas.models.imbalance.balanced_sample_weight``),
    not inside this pipeline builder.
    """
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classify",
                _ZeroIndexedXGBClassifier(
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.1,
                        objective="multi:softprob",
                        num_class=3,
                        random_state=42,
                        n_jobs=-1,
                        eval_metric="mlogloss",
                    )
                ),
            ),
        ]
    )


def build_lightgbm_pipeline(preprocessor, class_weight: str | dict | None = "balanced") -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classify",
                LGBMClassifier(
                    n_estimators=300,
                    class_weight=class_weight,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )


def build_catboost_pipeline(preprocessor, class_weights: list[float] | None = None) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classify",
                CatBoostClassifier(
                    iterations=300,
                    depth=6,
                    class_weights=class_weights,
                    random_state=42,
                    verbose=False,
                ),
            ),
        ]
    )
