from __future__ import annotations

import gc
import inspect
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from unfallatlas.models import candidate_analysis
from unfallatlas.models.artifacts import CandidateArtifact
from unfallatlas.models.candidate_analysis import (
    CandidateAnalysisResult,
    analysis_fingerprint,
    analyze_candidates,
    candidate_scores,
    compute_finalist_permutation_importance,
    load_or_analyze_candidates,
    measure_latency,
    measure_missing_feature_robustness,
    prediction_disagreement,
)


@pytest.fixture
def validation_data() -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    return pd.DataFrame(X, columns=["a", "b", "c", "d"]), pd.Series(y)


def _artifact(
    model: str,
    path: Path,
    *,
    sha256: str,
    family: str = "random_forest",
    evaluation_role: str = "finalist",
    score_interface: str = "predict_proba",
) -> CandidateArtifact:
    return CandidateArtifact(
        model=model,
        family=family,
        path=path,
        sha256=sha256,
        n_train=60,
        score_interface=score_interface,
        evaluation_role=evaluation_role,
    )


def test_candidate_scores_supports_probability_and_margin_models(validation_data):
    X, y = validation_data
    probability_model = LogisticRegression(max_iter=200).fit(X, y)
    margin_model = LinearSVC().fit(X, y)

    assert candidate_scores(probability_model, X).shape == (len(X),)
    assert candidate_scores(margin_model, X).shape == (len(X),)
    np.testing.assert_allclose(
        candidate_scores(probability_model, X),
        probability_model.predict_proba(X)[:, 1],
    )
    np.testing.assert_allclose(
        candidate_scores(margin_model, X),
        margin_model.decision_function(X),
    )


def test_candidate_scores_falls_back_to_predictions(validation_data):
    X, y = validation_data

    class PredictOnly:
        def __init__(self, predictions):
            self.predictions = predictions

        def predict(self, frame):
            return self.predictions[: len(frame)]

    model = PredictOnly(y.to_numpy())

    np.testing.assert_array_equal(candidate_scores(model, X), y.to_numpy(dtype=float))


def test_measure_latency_warms_up_and_returns_milliseconds_per_thousand_rows(
    validation_data, monkeypatch
):
    X, y = validation_data

    class CountingModel:
        def __init__(self):
            self.calls = 0

        def predict(self, frame):
            self.calls += 1
            return np.zeros(len(frame), dtype=int)

    times = iter([1.0, 1.002, 2.0, 2.004, 3.0, 3.006])
    monkeypatch.setattr(candidate_analysis.time, "perf_counter", lambda: next(times))
    model = CountingModel()

    latency = measure_latency(model, X.iloc[:20], repeats=3)

    assert model.calls == 4
    assert latency == pytest.approx(200.0)


def test_missing_feature_robustness_reports_measured_drift_and_class_changes(
    validation_data,
):
    X, y = validation_data
    model = make_pipeline(SimpleImputer(), LogisticRegression(max_iter=200)).fit(X, y)

    result = measure_missing_feature_robustness(model, X.iloc[:30], ["a"])

    assert set(result["a"]) >= {
        "prediction_failed",
        "mean_abs_score_drift",
        "changed_class_share",
    }
    assert result["a"]["prediction_failed"] is False
    assert result["a"]["mean_abs_score_drift"] >= 0.0
    assert 0.0 <= result["a"]["changed_class_share"] <= 1.0


def test_missing_feature_robustness_reports_prediction_failure(validation_data):
    X, y = validation_data
    model = LogisticRegression(max_iter=200).fit(X, y)

    result = measure_missing_feature_robustness(model, X.iloc[:30], ["a"])

    assert result["a"]["prediction_failed"] is True
    assert np.isnan(result["a"]["mean_abs_score_drift"])
    assert np.isnan(result["a"]["changed_class_share"])
    assert "NaN" in result["a"]["error"]


def test_missing_feature_robustness_records_non_value_prediction_errors(validation_data):
    X, _ = validation_data

    class RuntimeFailureModel:
        def _check(self, frame):
            if frame.isna().any().any():
                raise RuntimeError("candidate cannot score missing values")

        def predict_proba(self, frame):
            self._check(frame)
            scores = np.full(len(frame), 0.6)
            return np.column_stack([1.0 - scores, scores])

        def predict(self, frame):
            self._check(frame)
            return np.ones(len(frame), dtype=int)

    result = measure_missing_feature_robustness(RuntimeFailureModel(), X.iloc[:30], ["a"])

    assert result["a"]["prediction_failed"] is True
    assert result["a"]["error"] == "candidate cannot score missing values"


def test_analysis_uses_only_explicit_validation_inputs_and_releases_each_model(
    validation_data,
):
    X_val, y_val = validation_data
    artifacts = [
        _artifact("probability", Path("probability.joblib"), sha256="a" * 64),
        _artifact(
            "margin",
            Path("margin.joblib"),
            sha256="b" * 64,
            family="svm_linear",
            evaluation_role="candidate",
            score_interface="decision_function",
        ),
    ]
    fitted_models = {
        "probability.joblib": make_pipeline(SimpleImputer(), LogisticRegression(max_iter=200)).fit(
            X_val, y_val
        ),
        "margin.joblib": make_pipeline(SimpleImputer(), LinearSVC()).fit(X_val, y_val),
    }
    loaded = []

    def loader(path):
        loaded.append(path)
        return fitted_models[path.name]

    result = analyze_candidates(
        artifacts,
        X_val=X_val,
        y_val=y_val,
        latency_sample=X_val.iloc[:20],
        robustness_sample=X_val.iloc[:20],
        artifact_loader=loader,
    )

    assert loaded == [artifact.path for artifact in artifacts]
    assert len(result.metrics) == len(artifacts)
    assert set(result.scores) == {artifact.model for artifact in artifacts}
    assert set(result.predictions) == {artifact.model for artifact in artifacts}
    assert set(result.robustness["model"]) == {artifact.model for artifact in artifacts}
    assert not any(
        "test" in name.lower() for name in inspect.signature(analyze_candidates).parameters
    )


def test_analysis_releases_model_before_loading_next_candidate(validation_data):
    X_val, y_val = validation_data
    artifacts = [
        _artifact("one", Path("one.joblib"), sha256="1" * 64),
        _artifact("two", Path("two.joblib"), sha256="2" * 64),
    ]
    live_models = 0

    class EphemeralModel:
        def __init__(self):
            nonlocal live_models
            live_models += 1

        def __del__(self):
            nonlocal live_models
            live_models -= 1

        def predict_proba(self, frame):
            scores = np.full(len(frame), 0.6)
            return np.column_stack([1.0 - scores, scores])

        def predict(self, frame):
            return np.ones(len(frame), dtype=int)

    def loader(_path):
        gc.collect()
        assert live_models == 0
        return EphemeralModel()

    analyze_candidates(
        artifacts,
        X_val=X_val,
        y_val=y_val,
        latency_sample=X_val.iloc[:10],
        robustness_sample=X_val.iloc[:10],
        artifact_loader=loader,
        latency_repeats=1,
        robustness_columns=[],
    )

    gc.collect()
    assert live_models == 0


def test_analysis_loads_joblib_pipeline_by_default(tmp_path, validation_data):
    X_val, y_val = validation_data
    path = tmp_path / "pipeline.joblib"
    joblib.dump(
        make_pipeline(SimpleImputer(), LogisticRegression(max_iter=200)).fit(X_val, y_val),
        path,
    )
    artifacts = [_artifact("pipeline", path, sha256="a" * 64)]

    result = analyze_candidates(
        artifacts,
        X_val=X_val,
        y_val=y_val,
        latency_sample=X_val.iloc[:10],
        robustness_sample=X_val.iloc[:10],
        latency_repeats=1,
        robustness_columns=[],
    )

    assert result.metrics.loc[0, "model"] == "pipeline"
    assert len(result.scores["pipeline"]) == len(X_val)


def test_analysis_fingerprint_is_stable_and_tracks_every_input(tmp_path):
    artifacts = [
        _artifact("b", tmp_path / "b.joblib", sha256="b" * 64),
        _artifact("a", tmp_path / "a.joblib", sha256="a" * 64),
    ]
    parameters = {"random_state": 42, "latency_repeats": 5}

    fingerprint = analysis_fingerprint(artifacts, "data-v1", parameters)

    assert fingerprint == analysis_fingerprint(
        artifacts, "data-v1", dict(reversed(parameters.items()))
    )
    assert fingerprint != analysis_fingerprint(
        [replace(artifacts[0], sha256="c" * 64), artifacts[1]],
        "data-v1",
        parameters,
    )
    assert fingerprint != analysis_fingerprint(artifacts, "data-v2", parameters)
    assert fingerprint != analysis_fingerprint(
        artifacts,
        "data-v1",
        {"random_state": 42, "latency_repeats": 3},
    )


def test_load_or_analyze_recomputes_after_artifact_fingerprint_change(
    tmp_path, validation_data, monkeypatch
):
    X_val, y_val = validation_data
    artifacts = [_artifact("model", tmp_path / "model.joblib", sha256="a" * 64)]
    expected_result = CandidateAnalysisResult(
        metrics=pd.DataFrame(
            [
                {
                    "model": "model",
                    "family": "random_forest",
                    "evaluation_role": "finalist",
                    "macro_f1": 0.7,
                    "recall_ksi": 0.8,
                    "recall_slight": 0.6,
                    "latency_ms_per_1k": 2.0,
                }
            ]
        ),
        scores={"model": np.array([0.2, 0.8])},
        predictions={"model": np.array([0, 1])},
        robustness=pd.DataFrame(
            [
                {
                    "model": "model",
                    "feature": "a",
                    "prediction_failed": False,
                    "mean_abs_score_drift": 0.1,
                    "changed_class_share": 0.0,
                    "error": None,
                }
            ]
        ),
    )
    calls = []
    monkeypatch.setattr(
        candidate_analysis,
        "analyze_candidates",
        lambda *args, **kwargs: calls.append("computed") or expected_result,
    )
    parameters = {"latency_repeats": 5, "random_state": 42}

    load_or_analyze_candidates(
        artifacts,
        X_val=X_val,
        y_val=y_val,
        cache_dir=tmp_path,
        data_fingerprint="data-v1",
        parameters=parameters,
    )
    changed = [replace(artifacts[0], sha256="different")]
    load_or_analyze_candidates(
        changed,
        X_val=X_val,
        y_val=y_val,
        cache_dir=tmp_path,
        data_fingerprint="data-v1",
        parameters=parameters,
    )

    assert calls == ["computed", "computed"]


def test_load_or_analyze_reuses_matching_cache(tmp_path, validation_data, monkeypatch):
    X_val, y_val = validation_data
    artifacts = [_artifact("model", tmp_path / "model.joblib", sha256="a" * 64)]
    expected = CandidateAnalysisResult(
        metrics=pd.DataFrame([{"model": "model", "macro_f1": 0.7}]),
        scores={"model": np.array([0.2, 0.8])},
        predictions={"model": np.array([0, 1])},
        robustness=pd.DataFrame(
            [
                {
                    "model": "model",
                    "feature": "a",
                    "prediction_failed": False,
                    "mean_abs_score_drift": 0.1,
                    "changed_class_share": 0.0,
                    "error": None,
                }
            ]
        ),
    )
    calls = []
    monkeypatch.setattr(
        candidate_analysis,
        "analyze_candidates",
        lambda *args, **kwargs: calls.append("computed") or expected,
    )
    kwargs = {
        "X_val": X_val,
        "y_val": y_val,
        "cache_dir": tmp_path,
        "data_fingerprint": "data-v1",
        "parameters": {"latency_repeats": 5},
    }

    load_or_analyze_candidates(artifacts, **kwargs)
    cached = load_or_analyze_candidates(artifacts, **kwargs)

    assert calls == ["computed"]
    pd.testing.assert_frame_equal(cached.metrics, expected.metrics, check_dtype=False)
    np.testing.assert_allclose(cached.scores["model"], expected.scores["model"])
    np.testing.assert_array_equal(cached.predictions["model"], expected.predictions["model"])
    assert (tmp_path / "c_phase_analysis_manifest.json").is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_prediction_disagreement_returns_each_pair_once():
    predictions = {
        "a": np.array([0, 0, 1, 1]),
        "b": np.array([0, 1, 1, 0]),
        "c": np.array([1, 1, 1, 1]),
    }

    result = prediction_disagreement(predictions)

    assert list(zip(result["model_a"], result["model_b"], strict=True)) == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]
    assert result.loc[0, "disagreement_share"] == pytest.approx(0.5)


def test_prediction_disagreement_rejects_different_lengths():
    with pytest.raises(ValueError, match="same number"):
        prediction_disagreement({"a": np.array([0]), "b": np.array([0, 1])})


def test_finalist_permutation_importance_is_tidy_and_uses_fixed_sample(
    validation_data,
):
    X_val, y_val = validation_data
    artifacts = [
        _artifact("finalist", Path("finalist.joblib"), sha256="a" * 64),
        _artifact(
            "baseline",
            Path("baseline.joblib"),
            sha256="b" * 64,
            family="binary_random_guess",
            evaluation_role="baseline",
        ),
    ]
    fitted_models = {
        "finalist.joblib": make_pipeline(SimpleImputer(), LogisticRegression(max_iter=200)).fit(
            X_val, y_val
        ),
        "baseline.joblib": make_pipeline(SimpleImputer(), LogisticRegression(max_iter=200)).fit(
            X_val, y_val
        ),
    }
    loaded = []

    result = compute_finalist_permutation_importance(
        artifacts,
        X_val=X_val,
        y_val=y_val,
        sample_size=40,
        artifact_loader=lambda path: loaded.append(path.name) or fitted_models[path.name],
    )

    assert loaded == ["finalist.joblib"]
    assert list(result.columns) == [
        "model",
        "feature",
        "importance_mean",
        "importance_std",
        "rank",
    ]
    assert set(result["feature"]) == set(X_val.columns)
    assert result["rank"].min() == 1
    assert not any(
        "test" in name.lower()
        for name in inspect.signature(compute_finalist_permutation_importance).parameters
    )
