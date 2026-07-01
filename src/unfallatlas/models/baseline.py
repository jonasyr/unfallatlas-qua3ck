"""Baseline models — Stufe 0 of the A³ roadmap (docs/project/PROJEKTPLAN_SETUP.md)."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_random_guess_classifier() -> DummyClassifier:
    """Uniform random class guess — the theoretical macro-F1 floor (~0.33)."""
    return DummyClassifier(strategy="uniform", random_state=42)


def build_majority_class_classifier() -> DummyClassifier:
    """Always predicts the majority class (3 = Leicht) — exposes the imbalance problem."""
    return DummyClassifier(strategy="most_frequent")


def build_logreg_pipeline(preprocessor) -> Pipeline:
    """Logistic Regression baseline — the first non-trivial benchmark.

    ``preprocessor`` must come from ``build_preprocessor(scale_for_linear=True)``:
    LON/LAT and the binary transport-mode flags need scaling for a linear model.
    """
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classify",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
