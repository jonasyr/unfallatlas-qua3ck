import hashlib

import numpy as np
import pandas as pd
import pytest

from unfallatlas.models.artifacts import (
    build_candidate_registry,
    validate_candidate_registry,
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_build_candidate_registry_maps_every_comparison_row(tmp_path):
    binary_dir = tmp_path / "binary"
    binary_dir.mkdir()
    for model in ("binary_random_guess", "binary_random_forest_balanced"):
        (binary_dir / f"{model}.joblib").write_bytes(model.encode())
    comparison = pd.DataFrame(
        [
            {"model": "binary_random_guess", "family": "binary_random_guess", "n_train": np.nan},
            {
                "model": "binary_random_forest_balanced",
                "family": "random_forest",
                "n_train": 100,
            },
        ]
    )

    registry = build_candidate_registry(tmp_path, comparison)

    assert set(registry["candidates"]) == {
        "binary_random_guess",
        "binary_random_forest_balanced",
    }
    assert registry["candidates"]["binary_random_forest_balanced"]["score_interface"] == (
        "predict_proba"
    )


def test_validate_candidate_registry_rejects_fingerprint_drift(tmp_path):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"new")
    registry = {
        "candidates": {
            "m": {
                "path": "model.joblib",
                "sha256": sha256_bytes(b"old"),
                "family": "random_forest",
                "n_train": 10,
                "score_interface": "predict_proba",
                "evaluation_role": "finalist",
            }
        }
    }

    with pytest.raises(ValueError, match="fingerprint"):
        validate_candidate_registry(registry, tmp_path)


def test_build_registry_marks_supplied_winner_and_validates_relative_path(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    binary_dir = tmp_path / "data" / "checkpoints" / "run" / "binary"
    binary_dir.mkdir(parents=True)
    artifact = binary_dir / "binary_random_forest_balanced.joblib"
    artifact.write_bytes(b"forest")
    comparison = pd.DataFrame(
        [
            {
                "model": "binary_random_forest_balanced",
                "family": "random_forest",
                "n_train": 100.0,
            }
        ]
    )

    registry = build_candidate_registry(
        binary_dir.parent, comparison, champion_model="binary_random_forest_balanced"
    )
    artifacts = validate_candidate_registry(registry, tmp_path)

    assert registry["candidates"]["binary_random_forest_balanced"]["path"] == (
        "data/checkpoints/run/binary/binary_random_forest_balanced.joblib"
    )
    assert artifacts[0].path == artifact
    assert artifacts[0].evaluation_role == "champion"
    assert artifacts[0].n_train == 100


def test_validate_candidate_registry_rejects_unhashable_score_interface(tmp_path):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"model")
    registry = {
        "candidates": {
            "m": {
                "path": "model.joblib",
                "sha256": sha256_bytes(b"model"),
                "family": "random_forest",
                "n_train": 10,
                "score_interface": ["predict_proba"],
                "evaluation_role": "finalist",
            }
        }
    }

    with pytest.raises(ValueError, match="score interface"):
        validate_candidate_registry(registry, tmp_path)
