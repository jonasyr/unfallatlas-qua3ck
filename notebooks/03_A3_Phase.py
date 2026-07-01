# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: unfallatlas-qua3ck
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Unfallatlas Deutschland — A³-Phase
#
# **Phase:** Algorithm / Adapt / Adjust (A³) · 3 of 5 · QUA³CK
# **Goal of this notebook:** implement the U-phase §10 preprocessing contract,
# train and compare baseline and tree-ensemble models, select and tune the
# best imbalance-aware configuration, and report exactly one held-out
# test-2024 evaluation against the Q-phase acceptance criteria.
#
# **Strict scope.** This notebook does not compute SHAP values, does not
# discuss literature or limitations, and does not touch the Streamlit app —
# those are Phase C and Phase K. See
# `docs/superpowers/plans/2026-07-01-a3-phase-modelling.md` for the full
# scope boundary.
#
# ---

# %% [markdown]
# ## Position in the QUA³CK process
#
# | Phase | Notebook | Status |
# |:---|:---|:---:|
# | Q — Question | `01_Q_Phase.ipynb` | ✓ |
# | U — Understanding | `02_U_Phase.ipynb` | ✓ |
# | **A³ — Algorithm / Adapt / Adjust** | `03_A3_Phase.ipynb` | **→ here** |
# | C — Conclude & Compare | `04_C_Phase.ipynb` | pending |
# | K — Knowledge Transfer | `app/streamlit_app.py` | pending |
#
# ---

# %% [markdown]
# ## 0 — Setup and reproducibility

# %%
import subprocess
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import GroupKFold

from unfallatlas.features.preprocessing import (
    build_preprocessor,
    chronological_split,
    load_training_frame,
    split_features_target,
)
from unfallatlas.models.baseline import (
    build_logreg_pipeline,
    build_majority_class_classifier,
    build_random_guess_classifier,
)
from unfallatlas.models.boosting import (
    build_catboost_pipeline,
    build_lightgbm_pipeline,
    build_random_forest_pipeline,
    build_xgboost_pipeline,
    gpu_available,
)
from unfallatlas.models.evaluate import evaluate_predictions
from unfallatlas.models.imbalance import balanced_sample_weight

# %%
pio.templates.default = "plotly_white"
pio.renderers.default = "vscode"
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
np.random.seed(42)

BASE_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FIG_DIR = BASE_DIR / "reports" / "figures" / "a3_phase"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _git_short_sha():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# %% [markdown]
# ### GPU acceleration (optional, machine-specific — not part of the reproducible contract)
#
# `USE_GPU` controls XGBoost/LightGBM/CatBoost's training device:
# - `None` (default) — auto-detect via `nvidia-smi`; falls back to CPU with
#   no code change on a machine without a CUDA GPU (e.g. a grader's).
# - `True` — force GPU (fails loudly if no compatible GPU/driver is present).
# - `False` — force CPU everywhere, even if a GPU is available.
#
# Random Forest and Logistic Regression always run on CPU (scikit-learn has
# no GPU backend without a separate RAPIDS/cuML environment, which this
# project does not depend on).

# %%
USE_GPU = None  # None = auto-detect, True = force GPU, False = force CPU

_use_gpu_resolved = gpu_available() if USE_GPU is None else USE_GPU
print(f"GPU acceleration: {'ON' if _use_gpu_resolved else 'OFF'}  (USE_GPU={USE_GPU})")

# %% [markdown]
# ### Progress logging and per-model checkpointing
#
# `nbconvert --execute` does not stream cell output live and only writes the
# `.ipynb` file once the whole run finishes (or crashes) — so without an
# explicit external log, there is no way to see progress while a long cell is
# running, and no way to recover already-fitted models if a later cell fails.
#
# `_log_progress()` appends timestamped lines directly to
# `reports/a3_progress.log` (flushed immediately, so `tail -f` shows it live).
# `_fit_or_checkpoint()` saves each fitted pipeline to
# `data/processed/a3_checkpoints/` right after training and reloads from there
# on a re-run instead of refitting — so a crash in, say, CatBoost training does
# not require redoing Random Forest/XGBoost/LightGBM.

# %%
PROGRESS_LOG = BASE_DIR / "reports" / "a3_progress.log"
PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = PROCESSED_DIR / "a3_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def _log_progress(message: str) -> None:
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    line = f"[{timestamp}] {message}"
    print(line)
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")
        f.flush()


def _fit_or_checkpoint(name: str, build_fn, fit_kwargs: dict | None = None):
    """Load a cached fitted pipeline for `name` if present, else fit + save it.

    Lets a re-run after a crash skip every model that already finished,
    rather than refitting everything from scratch.
    """
    checkpoint_path = CHECKPOINT_DIR / f"{name}.joblib"
    if checkpoint_path.exists():
        _log_progress(f"  -> {name}: loaded from checkpoint ({checkpoint_path.name})")
        return joblib.load(checkpoint_path)
    pipeline = build_fn()
    if fit_kwargs:
        pipeline.fit(X_train, y_train, **fit_kwargs)
    else:
        pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, checkpoint_path)
    return pipeline


def run_stage(
    name: str, build_fn, fit_kwargs: dict | None, index: int, total: int, durations: list[float]
):
    """Fit-or-load one model with progress/ETA logging, then score it on validation."""
    pct = 100 * (index - 1) / total
    if durations:
        eta_seconds = (sum(durations) / len(durations)) * (total - index + 1)
        eta_str = f"{eta_seconds / 60:.1f} min"
    else:
        eta_str = "unknown"
    _log_progress(f"[{index}/{total}] ({pct:.0f}%) training {name} ... ETA remaining: {eta_str}")
    start = time.time()
    pipeline = _fit_or_checkpoint(name, build_fn, fit_kwargs)
    elapsed = time.time() - start
    durations.append(elapsed)
    _score_on_validation(name, pipeline)
    _log_progress(f"  -> {name} done in {elapsed:.1f}s")
    return pipeline


# %% [markdown]
# ## 1 — Load the U-phase cache and apply the chronological split
#
# A³ does not rebuild the DWD-enriched cache — it reads exactly what
# `notebooks/02_U_Phase.ipynb` §8.5 already produced.

# %%
_log_progress("Loading U-phase weather-enriched cache and applying chronological split...")
df = load_training_frame(BASE_DIR)
train_df, val_df, test_df = chronological_split(df)

X_train, y_train = split_features_target(train_df)
X_val, y_val = split_features_target(val_df)
X_test, y_test = split_features_target(test_df)

provenance = {
    "rows_train": len(X_train),
    "rows_val": len(X_val),
    "rows_test": len(X_test),
    "git_commit": _git_short_sha(),
    "run_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "random_seed": 42,
}
for k, v in provenance.items():
    print(f"  {k:14s} {v}")
_log_progress(
    f"Data loaded: {provenance['rows_train']:,} train / {provenance['rows_val']:,} val / {provenance['rows_test']:,} test rows."
)

# %% [markdown]
# ## 2 — Cross-validation strategy
#
# U-phase §10: "either a chronological TimeSeriesSplit or a year-grouped
# K-fold; not a random StratifiedKFold." The training window spans 7
# distinct years (2016–2022) — `GroupKFold` grouped by year gives 7 folds
# that never let a model see a "future" year during model selection.

# %%
cv_groups = train_df["UJAHR"].to_numpy()
cv = GroupKFold(n_splits=train_df["UJAHR"].nunique())
print(f"GroupKFold with {cv.get_n_splits(groups=cv_groups)} year-groups (2016-2022).")

# %% [markdown]
# ## 3 — Stufe 0: baselines
#
# Random guess and majority class establish the macro-F1 floor the Q-phase
# expects (~0.33 and ~0.30 respectively); Logistic Regression is the first
# non-trivial benchmark.

# %%
tree_preprocessor = build_preprocessor(scale_for_linear=False)
linear_preprocessor = build_preprocessor(scale_for_linear=True)

comparison_rows: list[dict] = []


def _score_on_validation(name: str, fitted_estimator) -> None:
    preds = fitted_estimator.predict(X_val)
    metrics = evaluate_predictions(y_val, preds)
    comparison_rows.append({"model": name, **metrics})
    print(
        f"{name:35s} macro-F1={metrics['macro_f1']:.3f}  recall(1)={metrics['recall_class_1']:.3f}"
    )


stufe0_specs = [
    ("random_guess", lambda: build_random_guess_classifier(), None),
    ("majority_class", lambda: build_majority_class_classifier(), None),
    ("logistic_regression", lambda: build_logreg_pipeline(linear_preprocessor), None),
]

fitted_baselines: dict = {}
_stufe0_durations: list[float] = []
_log_progress(f"Starting Stufe 0: {len(stufe0_specs)} baselines.")
for i, (name, build_fn, fit_kwargs) in enumerate(stufe0_specs, start=1):
    fitted_baselines[name] = run_stage(
        name, build_fn, fit_kwargs, index=i, total=len(stufe0_specs), durations=_stufe0_durations
    )
_log_progress(
    f"Stufe 0 complete: {len(stufe0_specs)}/{len(stufe0_specs)} (100%) baselines trained."
)

random_guess = fitted_baselines["random_guess"]
majority_class = fitted_baselines["majority_class"]
logreg = fitted_baselines["logistic_regression"]

# %% [markdown]
# ## 4 — Stufe 1: tree ensembles, default and class-weighted
#
# Each of Random Forest, XGBoost, LightGBM, and CatBoost is trained twice:
# once with library defaults, once with class-weighting applied — 8
# configurations, all scored once against the 2023 validation split.

# %%
train_class_counts = y_train.value_counts()
catboost_weights = [len(y_train) / (3 * train_class_counts[c]) for c in [1, 2, 3]]
xgb_weights = balanced_sample_weight(y_train)

stufe1_specs = [
    (
        "random_forest_default",
        lambda: build_random_forest_pipeline(tree_preprocessor, class_weight=None),
        None,
    ),
    (
        "random_forest_balanced",
        lambda: build_random_forest_pipeline(tree_preprocessor, class_weight="balanced"),
        None,
    ),
    (
        "xgboost_default",
        lambda: build_xgboost_pipeline(tree_preprocessor, use_gpu=_use_gpu_resolved),
        None,
    ),
    (
        "xgboost_balanced",
        lambda: build_xgboost_pipeline(tree_preprocessor, use_gpu=_use_gpu_resolved),
        {"classify__sample_weight": xgb_weights},
    ),
    (
        "lightgbm_default",
        lambda: build_lightgbm_pipeline(
            tree_preprocessor, class_weight=None, use_gpu=_use_gpu_resolved
        ),
        None,
    ),
    (
        "lightgbm_balanced",
        lambda: build_lightgbm_pipeline(
            tree_preprocessor, class_weight="balanced", use_gpu=_use_gpu_resolved
        ),
        None,
    ),
    (
        "catboost_default",
        lambda: build_catboost_pipeline(tree_preprocessor, use_gpu=_use_gpu_resolved),
        None,
    ),
    (
        "catboost_balanced",
        lambda: build_catboost_pipeline(
            tree_preprocessor, class_weights=catboost_weights, use_gpu=_use_gpu_resolved
        ),
        None,
    ),
]

fitted_models: dict = {}
_stufe1_durations: list[float] = []
_log_progress(f"Starting Stufe 1: {len(stufe1_specs)} tree-ensemble configurations.")
for i, (name, build_fn, fit_kwargs) in enumerate(stufe1_specs, start=1):
    fitted_models[name] = run_stage(
        name, build_fn, fit_kwargs, index=i, total=len(stufe1_specs), durations=_stufe1_durations
    )
_log_progress(f"Stufe 1 complete: {len(stufe1_specs)}/{len(stufe1_specs)} (100%) models trained.")

# %% [markdown]
# ## 5 — Champion selection
#
# The champion is the single highest validation macro-F1 among the 8 tree
# configurations above (baselines are never candidates for champion — they
# exist to bound the floor, not to compete for it).

# %%
comparison_df = pd.DataFrame(comparison_rows)
tree_only = comparison_df[comparison_df["model"].isin(fitted_models.keys())]
champion_name = tree_only.sort_values("macro_f1", ascending=False).iloc[0]["model"]
champion_pipeline = fitted_models[champion_name]
print(f"Champion (highest validation macro-F1 among tree configurations): {champion_name}")
comparison_df.sort_values("macro_f1", ascending=False)

# %%
fig = px.bar(
    comparison_df.sort_values("macro_f1"),
    x="macro_f1",
    y="model",
    orientation="h",
    title="Validation macro-F1 by model (Stufe 0 + Stufe 1)",
)
fig.add_vline(x=0.55, line_dash="dash", annotation_text="Q-phase threshold (0.55)")
fig.write_html(FIG_DIR / "05_model_comparison_stufe0_1.html", include_plotlyjs="cdn")
fig.show()

# %% [markdown]
# > **Transition.** The champion architecture is selected on validation
# > macro-F1. Part 2 of this notebook compares imbalance strategies on the
# > champion only, tunes the winner, and reports the single test-2024
# > evaluation.
