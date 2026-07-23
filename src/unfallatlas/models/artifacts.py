"""Persisted model-candidate registry helpers for the C phase."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_ARTIFACT_KEYS = {
    "path",
    "sha256",
    "family",
    "n_train",
    "score_interface",
    "evaluation_role",
}
_FINALIST_FAMILIES = {"random_forest", "xgboost", "lightgbm", "catboost"}
_DECISION_FUNCTION_FAMILIES = {"svm_linear", "svm_sgd", "svm_rbf"}
_SCORE_INTERFACES = {"predict_proba", "decision_function", "predict"}
_EVALUATION_ROLES = {"baseline", "candidate", "finalist", "champion"}
_BASELINE_MODELS = {"binary_random_guess", "binary_majority_class"}


@dataclass(frozen=True)
class CandidateArtifact:
    """A validated, persisted binary-model candidate."""

    model: str
    family: str
    path: Path
    sha256: str
    n_train: int | None
    score_interface: Literal["predict_proba", "decision_function", "predict"]
    evaluation_role: Literal["baseline", "candidate", "finalist", "champion"]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 fingerprint of ``path`` without loading its model."""
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_candidate_registry(
    checkpoint_dir: Path, comparison_df, champion_model: str | None = None
) -> dict:
    """Build serialisable metadata for every binary checkpoint in a comparison table."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    repo_root = _find_repository_root(checkpoint_dir)
    candidates: dict[str, dict] = {}

    for row in comparison_df.to_dict(orient="records"):
        model = row["model"]
        if model in candidates:
            raise ValueError(f"Comparison table contains duplicate model {model!r}.")

        family = row["family"]
        artifact_path = checkpoint_dir / "binary" / f"{model}.joblib"
        candidates[model] = {
            "path": str(artifact_path.resolve().relative_to(repo_root)),
            "sha256": sha256_file(artifact_path),
            "family": family,
            "n_train": _normalise_n_train(row.get("n_train")),
            "score_interface": _score_interface_for(family),
            "evaluation_role": _evaluation_role_for(model, family, champion_model),
        }

    if champion_model is not None and champion_model not in candidates:
        raise ValueError(f"Champion model {champion_model!r} is not in the comparison table.")

    return {"candidates": candidates}


def validate_candidate_registry(registry: dict, repo_root: Path) -> list[CandidateArtifact]:
    """Validate registry metadata and return load-ready candidate artifacts."""
    if set(registry) != {"candidates"}:
        raise ValueError("Registry must contain exactly the 'candidates' key.")
    if not isinstance(registry["candidates"], dict):
        raise ValueError("Registry candidates must be a mapping.")

    repo_root = Path(repo_root).resolve()
    artifacts: list[CandidateArtifact] = []
    for model, payload in registry["candidates"].items():
        if not isinstance(model, str) or not model:
            raise ValueError("Candidate model names must be non-empty strings.")
        if not isinstance(payload, dict) or set(payload) != _ARTIFACT_KEYS:
            raise ValueError(f"Candidate {model!r} has invalid registry keys.")

        path = _validated_path(payload["path"], repo_root, model)
        _validate_payload(payload, model)
        observed_sha256 = sha256_file(path)
        if observed_sha256 != payload["sha256"]:
            raise ValueError(f"Candidate {model!r} fingerprint does not match its artifact.")

        artifacts.append(
            CandidateArtifact(
                model=model,
                family=payload["family"],
                path=path,
                sha256=payload["sha256"],
                n_train=payload["n_train"],
                score_interface=payload["score_interface"],
                evaluation_role=payload["evaluation_role"],
            )
        )

    return artifacts


def _find_repository_root(checkpoint_dir: Path) -> Path:
    for directory in (checkpoint_dir, *checkpoint_dir.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    return checkpoint_dir


def _normalise_n_train(value) -> int | None:
    if value is None:
        return None
    try:
        if math.isnan(value):
            return None
    except TypeError:
        pass
    return int(value)


def _score_interface_for(family: str) -> Literal["predict_proba", "decision_function", "predict"]:
    if family in _DECISION_FUNCTION_FAMILIES:
        return "decision_function"
    return "predict_proba"


def _evaluation_role_for(
    model: str, family: str, champion_model: str | None
) -> Literal["baseline", "candidate", "finalist", "champion"]:
    if model == champion_model:
        return "champion"
    if family in _FINALIST_FAMILIES:
        return "finalist"
    if model in _BASELINE_MODELS:
        return "baseline"
    return "candidate"


def _validated_path(path_value, repo_root: Path, model: str) -> Path:
    if not isinstance(path_value, str):
        raise ValueError(f"Candidate {model!r} path must be a relative string.")
    relative_path = Path(path_value)
    if relative_path.is_absolute():
        raise ValueError(f"Candidate {model!r} path must stay inside the repository.")

    path = (repo_root / relative_path).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as error:
        raise ValueError(f"Candidate {model!r} path escapes the repository.") from error
    if not path.is_file():
        raise ValueError(f"Candidate {model!r} artifact does not exist: {relative_path}")
    return path


def _validate_payload(payload: dict, model: str) -> None:
    if not isinstance(payload["family"], str) or not payload["family"]:
        raise ValueError(f"Candidate {model!r} family must be a non-empty string.")
    if not isinstance(payload["sha256"], str) or len(payload["sha256"]) != 64:
        raise ValueError(f"Candidate {model!r} has an invalid SHA-256 fingerprint.")
    try:
        int(payload["sha256"], 16)
    except ValueError as error:
        raise ValueError(f"Candidate {model!r} has an invalid SHA-256 fingerprint.") from error
    if payload["n_train"] is not None and (
        isinstance(payload["n_train"], bool) or not isinstance(payload["n_train"], int)
    ):
        raise ValueError(f"Candidate {model!r} n_train must be an integer or null.")
    if (
        not isinstance(payload["score_interface"], str)
        or payload["score_interface"] not in _SCORE_INTERFACES
    ):
        raise ValueError(f"Candidate {model!r} has an unsupported score interface.")
    if (
        not isinstance(payload["evaluation_role"], str)
        or payload["evaluation_role"] not in _EVALUATION_ROLES
    ):
        raise ValueError(f"Candidate {model!r} has an unsupported evaluation role.")
