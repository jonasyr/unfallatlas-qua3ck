"""Validation-only evidence for persisted C-phase model candidates."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedShuffleSplit

from unfallatlas.models.artifacts import CandidateArtifact
from unfallatlas.models.evaluate import evaluate_binary_predictions

_MANIFEST_NAME = "c_phase_analysis_manifest.json"
_METRICS_NAME = "c_phase_candidate_metrics.csv"
_SCORES_NAME = "c_phase_candidate_scores.parquet"
_ROBUSTNESS_NAME = "c_phase_candidate_robustness.csv"
_FINALIST_ROLES = {"finalist", "champion"}


@dataclass
class CandidateAnalysisResult:
    """Compact validation evidence retained after candidate models are released."""

    metrics: pd.DataFrame
    scores: dict[str, np.ndarray]
    predictions: dict[str, np.ndarray]
    robustness: pd.DataFrame


def candidate_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Return a one-dimensional positive-class score for a binary candidate."""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X))
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError("predict_proba must return two binary-class columns.")
        return probabilities[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).reshape(-1)
    return np.asarray(model.predict(X), dtype=float).reshape(-1)


def measure_latency(model, X: pd.DataFrame, repeats: int = 5) -> float:
    """Measure warmed-up median prediction latency in milliseconds per 1,000 rows."""
    if repeats < 1:
        raise ValueError("repeats must be at least one.")
    if len(X) == 0:
        raise ValueError("The latency sample must not be empty.")

    model.predict(X)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(X)
        elapsed.append(time.perf_counter() - start)
    return float(np.median(elapsed) * 1_000_000 / len(X))


def measure_missing_feature_robustness(
    model,
    X: pd.DataFrame,
    columns: Sequence[str],
) -> dict[str, dict]:
    """Measure score and class drift when each selected feature is missing."""
    missing_columns = [column for column in columns if column not in X.columns]
    if missing_columns:
        raise ValueError(f"Robustness columns are absent from the sample: {missing_columns}")
    if not columns:
        return {}

    baseline_scores = candidate_scores(model, X)
    baseline_predictions = np.asarray(model.predict(X)).reshape(-1)
    results: dict[str, dict] = {}
    for column in columns:
        perturbed = X.copy()
        perturbed[column] = np.nan
        try:
            perturbed_scores = candidate_scores(model, perturbed)
            perturbed_predictions = np.asarray(model.predict(perturbed)).reshape(-1)
        except Exception as error:
            results[column] = {
                "prediction_failed": True,
                "mean_abs_score_drift": np.nan,
                "changed_class_share": np.nan,
                "error": str(error),
            }
            continue

        results[column] = {
            "prediction_failed": False,
            "mean_abs_score_drift": float(np.mean(np.abs(perturbed_scores - baseline_scores))),
            "changed_class_share": float(np.mean(perturbed_predictions != baseline_predictions)),
            "error": None,
        }
    return results


def analyze_candidates(
    artifacts: Sequence[CandidateArtifact],
    *,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    latency_sample: pd.DataFrame,
    robustness_sample: pd.DataFrame,
    artifact_loader: Callable[[Path], object] = joblib.load,
    latency_repeats: int = 5,
    robustness_columns: Sequence[str] | None = None,
) -> CandidateAnalysisResult:
    """Analyze candidates sequentially using Val-2023 inputs only."""
    if len(X_val) != len(y_val):
        raise ValueError("X_val and y_val must contain the same number of rows.")
    probe_columns = (
        list(robustness_sample.columns) if robustness_columns is None else list(robustness_columns)
    )
    metric_rows: list[dict] = []
    robustness_rows: list[dict] = []
    scores: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}

    for artifact in artifacts:
        model = artifact_loader(artifact.path)
        try:
            model_scores = candidate_scores(model, X_val)
            model_predictions = np.asarray(model.predict(X_val)).reshape(-1)
            if len(model_scores) != len(X_val) or len(model_predictions) != len(X_val):
                raise ValueError(
                    f"Candidate {artifact.model!r} returned the wrong validation row count."
                )

            binary_metrics = evaluate_binary_predictions(y_val, model_predictions)
            metric_rows.append(
                {
                    "model": artifact.model,
                    "family": artifact.family,
                    "evaluation_role": artifact.evaluation_role,
                    "macro_f1": binary_metrics["macro_f1"],
                    "recall_ksi": binary_metrics["recall_ksi"],
                    "recall_slight": binary_metrics["recall_slight"],
                    "confusion_matrix": json.dumps(
                        binary_metrics["confusion_matrix"], separators=(",", ":")
                    ),
                    "latency_ms_per_1k": measure_latency(
                        model, latency_sample, repeats=latency_repeats
                    ),
                }
            )
            robustness = measure_missing_feature_robustness(model, robustness_sample, probe_columns)
            robustness_rows.extend(
                {"model": artifact.model, "feature": feature, **measurements}
                for feature, measurements in robustness.items()
            )
            scores[artifact.model] = np.asarray(model_scores).copy()
            predictions[artifact.model] = np.asarray(model_predictions).copy()
        finally:
            del model
            gc.collect()

    robustness_columns_order = [
        "model",
        "feature",
        "prediction_failed",
        "mean_abs_score_drift",
        "changed_class_share",
        "error",
    ]
    return CandidateAnalysisResult(
        metrics=pd.DataFrame(metric_rows),
        scores=scores,
        predictions=predictions,
        robustness=pd.DataFrame(robustness_rows, columns=robustness_columns_order),
    )


def analysis_fingerprint(
    artifacts: Sequence[CandidateArtifact],
    data_fingerprint: str,
    parameters: Mapping[str, object],
) -> str:
    """Hash ordered candidate identities, validation data, and analysis parameters."""
    payload = {
        "artifacts": [
            {"model": artifact.model, "sha256": artifact.sha256} for artifact in artifacts
        ],
        "data_fingerprint": data_fingerprint,
        "parameters": dict(parameters),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_or_analyze_candidates(
    artifacts: Sequence[CandidateArtifact],
    *,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cache_dir: Path,
    data_fingerprint: str,
    parameters: Mapping[str, object],
    latency_sample: pd.DataFrame | None = None,
    robustness_sample: pd.DataFrame | None = None,
    artifact_loader: Callable[[Path], object] = joblib.load,
) -> CandidateAnalysisResult:
    """Reuse fingerprint-matched evidence or recompute and atomically replace it."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = analysis_fingerprint(artifacts, data_fingerprint, parameters)
    paths = {
        "manifest": cache_dir / _MANIFEST_NAME,
        "metrics": cache_dir / _METRICS_NAME,
        "scores": cache_dir / _SCORES_NAME,
        "robustness": cache_dir / _ROBUSTNESS_NAME,
    }
    if _cache_matches(paths, fingerprint):
        return _read_cached_result(paths)

    result = analyze_candidates(
        artifacts,
        X_val=X_val,
        y_val=y_val,
        latency_sample=X_val if latency_sample is None else latency_sample,
        robustness_sample=X_val if robustness_sample is None else robustness_sample,
        artifact_loader=artifact_loader,
        latency_repeats=int(parameters.get("latency_repeats", 5)),
        robustness_columns=parameters.get("robustness_columns"),
    )
    _atomic_dataframe_write(result.metrics, paths["metrics"], format_name="csv")
    _atomic_dataframe_write(result.robustness, paths["robustness"], format_name="csv")
    _atomic_dataframe_write(_scores_frame(result), paths["scores"], format_name="parquet")
    _atomic_json_write(
        {
            "fingerprint": fingerprint,
            "artifacts": [
                {"model": artifact.model, "sha256": artifact.sha256} for artifact in artifacts
            ],
            "data_fingerprint": data_fingerprint,
            "parameters": dict(parameters),
        },
        paths["manifest"],
    )
    return result


def prediction_disagreement(predictions: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Return pairwise disagreement shares at the supplied validation operating points."""
    names = list(predictions)
    lengths = {len(np.asarray(predictions[name]).reshape(-1)) for name in names}
    if len(lengths) > 1:
        raise ValueError("All prediction arrays must contain the same number of rows.")

    rows = []
    for left_index, model_a in enumerate(names):
        left = np.asarray(predictions[model_a]).reshape(-1)
        for model_b in names[left_index + 1 :]:
            right = np.asarray(predictions[model_b]).reshape(-1)
            rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "disagreement_share": float(np.mean(left != right)),
                }
            )
    return pd.DataFrame(rows, columns=["model_a", "model_b", "disagreement_share"])


def compute_finalist_permutation_importance(
    artifacts: Sequence[CandidateArtifact],
    *,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_size: int,
    artifact_loader: Callable[[Path], object] = joblib.load,
    n_repeats: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute model-agnostic finalist importance on one fixed stratified Val sample."""
    X_sample, y_sample = _fixed_stratified_sample(
        X_val, y_val, sample_size=sample_size, random_state=random_state
    )
    rows: list[dict] = []
    for artifact in artifacts:
        if artifact.evaluation_role not in _FINALIST_ROLES:
            continue
        model = artifact_loader(artifact.path)
        try:
            importance = permutation_importance(
                model,
                X_sample,
                y_sample,
                scoring="f1_macro",
                n_repeats=n_repeats,
                random_state=random_state,
            )
            ranks = (
                pd.Series(importance.importances_mean)
                .rank(method="min", ascending=False)
                .astype(int)
                .to_numpy()
            )
            rows.extend(
                {
                    "model": artifact.model,
                    "feature": feature,
                    "importance_mean": float(mean),
                    "importance_std": float(std),
                    "rank": int(rank),
                }
                for feature, mean, std, rank in zip(
                    X_sample.columns,
                    importance.importances_mean,
                    importance.importances_std,
                    ranks,
                    strict=True,
                )
            )
        finally:
            del model
            gc.collect()
    return pd.DataFrame(
        rows,
        columns=["model", "feature", "importance_mean", "importance_std", "rank"],
    )


def _fixed_stratified_sample(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    sample_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X_val) != len(y_val):
        raise ValueError("X_val and y_val must contain the same number of rows.")
    if sample_size < 1:
        raise ValueError("sample_size must be at least one.")
    if sample_size >= len(X_val):
        return X_val.reset_index(drop=True), pd.Series(y_val).reset_index(drop=True)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=random_state)
    sample_indices, _ = next(splitter.split(X_val, y_val))
    return (
        X_val.iloc[sample_indices].reset_index(drop=True),
        pd.Series(y_val).iloc[sample_indices].reset_index(drop=True),
    )


def _cache_matches(paths: Mapping[str, Path], fingerprint: str) -> bool:
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("fingerprint") == fingerprint


def _scores_frame(result: CandidateAnalysisResult) -> pd.DataFrame:
    rows = []
    if set(result.scores) != set(result.predictions):
        raise ValueError("Scores and predictions must contain the same candidate models.")
    for model, model_scores in result.scores.items():
        model_predictions = np.asarray(result.predictions[model]).reshape(-1)
        model_scores = np.asarray(model_scores).reshape(-1)
        if len(model_scores) != len(model_predictions):
            raise ValueError(f"Scores and predictions differ in length for {model!r}.")
        rows.extend(
            {
                "model": model,
                "row": row,
                "score": float(score),
                "prediction": prediction,
            }
            for row, (score, prediction) in enumerate(
                zip(model_scores, model_predictions, strict=True)
            )
        )
    return pd.DataFrame(rows, columns=["model", "row", "score", "prediction"])


def _read_cached_result(paths: Mapping[str, Path]) -> CandidateAnalysisResult:
    metrics = pd.read_csv(paths["metrics"])
    robustness = pd.read_csv(paths["robustness"])
    score_frame = pd.read_parquet(paths["scores"]).sort_values(["model", "row"])
    scores = {
        model: group["score"].to_numpy()
        for model, group in score_frame.groupby("model", sort=False)
    }
    predictions = {
        model: group["prediction"].to_numpy()
        for model, group in score_frame.groupby("model", sort=False)
    }
    return CandidateAnalysisResult(
        metrics=metrics,
        scores=scores,
        predictions=predictions,
        robustness=robustness,
    )


def _atomic_dataframe_write(frame: pd.DataFrame, path: Path, *, format_name: str) -> None:
    with NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        if format_name == "csv":
            frame.to_csv(temporary_path, index=False)
        elif format_name == "parquet":
            frame.to_parquet(temporary_path, index=False)
        else:
            raise ValueError(f"Unsupported cache format {format_name!r}.")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json_write(payload: Mapping[str, object], path: Path) -> None:
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(payload, temporary, sort_keys=True, separators=(",", ":"))
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
