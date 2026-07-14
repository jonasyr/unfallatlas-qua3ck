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
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold, cross_validate

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
from unfallatlas.models.evaluate import (
    evaluate_predictions,
    meets_acceptance_criteria,
    recall_for_class,
    select_best_candidate,
)
from unfallatlas.models.imbalance import (
    balanced_sample_weight,
    find_best_threshold_for_class,
    resample_adasyn,
    resample_smote,
)
from unfallatlas.models.ordinal import build_ordinal_pipeline

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
# - `None` (default) — auto-detect CUDA for XGBoost and CatBoost. LightGBM
#   stays on CPU because its GPU mode requires a separately compatible OpenCL runtime.
# - `True` — force GPU; LightGBM fails loudly if no compatible OpenCL device is present.
# - `False` — force CPU everywhere, even if a GPU is available.
#
# Random Forest and Logistic Regression always run on CPU (scikit-learn has
# no GPU backend without a separate RAPIDS/cuML environment, which this
# project does not depend on).

# %%
USE_GPU = None  # Auto CUDA for XGBoost/CatBoost; CPU for LightGBM by default.

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
# `data/processed/a3_checkpoints/<git-commit>/` right after training and
# reloads from there on a re-run instead of refitting — so a crash in, say,
# CatBoost training does not require redoing Random Forest/XGBoost/LightGBM.
#
# The checkpoint directory is scoped by the current short git commit hash
# (uncommitted changes still share the last commit's directory — clear
# `data/processed/a3_checkpoints/` manually while actively iterating on
# uncommitted model-builder changes). Any *committed* change to a model
# builder's hyperparameters lands in a new commit, which gets a fresh,
# empty checkpoint directory automatically — a stale checkpoint from a
# previous configuration can never silently be loaded as if it reflected
# the current code.

# %%
PROGRESS_LOG = BASE_DIR / "reports" / "a3_progress.log"
PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = PROCESSED_DIR / "a3_checkpoints" / _git_short_sha()
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

    Checkpoints live under CHECKPOINT_DIR, which is scoped by the current
    git commit hash — so a committed change to any model builder's
    hyperparameters automatically gets a fresh cache directory instead of
    silently reusing a stale checkpoint from a different configuration.
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


def _extract_family(name: str) -> str:
    """'random_forest_balanced' -> 'random_forest'. Same-family runs
    (default/balanced) cost nearly identical time; different families can
    differ by 10x+ (e.g. Random Forest vs. GPU-accelerated CatBoost)."""
    return name.removesuffix("_default").removesuffix("_balanced")


def _eta_seconds(
    remaining_names: list[str],
    family_durations: dict[str, list[float]],
    last_duration: float | None,
) -> float | None:
    """Family-aware ETA.

    For each remaining stage, use its own family's observed average if
    we've already timed that family (default/balanced pairs cost almost
    the same). For a family not yet seen at all, fall back to the most
    recently observed single duration (of ANY family) rather than the
    overall historical average — so the estimate reacts within one step
    when training shifts from a slow family (Random Forest) to a fast one
    (XGBoost/LightGBM/CatBoost), instead of staying dragged down by the
    slow models seen earlier.
    """
    if last_duration is None:
        return None
    total = 0.0
    for name in remaining_names:
        family = _extract_family(name)
        family_seen = family_durations.get(family)
        total += (sum(family_seen) / len(family_seen)) if family_seen else last_duration
    return total


def run_stage(
    name: str,
    build_fn,
    fit_kwargs: dict | None,
    index: int,
    total: int,
    all_names: list[str],
    timing_state: dict,
):
    """Fit-or-load one model with family-aware progress/ETA logging, then score it.

    `timing_state` is a per-stage-group dict `{"family_durations": {}, "last_duration": None}`,
    shared and mutated across every call within one Stufe 0 / Stufe 1 loop
    (kept separate between the two loops since baselines and tree ensembles
    have unrelated cost profiles).
    """
    remaining_names = all_names[index:]  # stages after this one
    eta_seconds = _eta_seconds(
        remaining_names, timing_state["family_durations"], timing_state["last_duration"]
    )
    pct = 100 * (index - 1) / total
    eta_str = f"{eta_seconds / 60:.1f} min" if eta_seconds is not None else "unknown"
    _log_progress(f"[{index}/{total}] ({pct:.0f}%) training {name} ... ETA remaining: {eta_str}")
    start = time.time()
    pipeline = _fit_or_checkpoint(name, build_fn, fit_kwargs)
    elapsed = time.time() - start
    family = _extract_family(name)
    timing_state["family_durations"].setdefault(family, []).append(elapsed)
    timing_state["last_duration"] = elapsed
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
#
# **This cell is a standalone sanity check, not consumed downstream.** It
# confirms the CV strategy resolves to the expected 7 year-groups on the
# full training set. §7's Optuna tuning (below) builds its own
# `GroupKFold` from `years_sub` — the *500k-row subsample's* distinct
# years, which can differ in count from the full set's — so it
# intentionally does not reuse `cv`/`cv_groups` from this cell.

# %%
# Illustrative only - see the markdown note above for why §7 builds its
# own GroupKFold from the subsample's years instead of reusing this one.
_cv_groups_full_train = train_df["UJAHR"].to_numpy()
_cv_full_train = GroupKFold(n_splits=train_df["UJAHR"].nunique())
print(
    f"GroupKFold with {_cv_full_train.get_n_splits(groups=_cv_groups_full_train)} "
    "year-groups (2016-2022)."
)

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


def _score_on_validation(name: str, fitted_estimator, X_val_override=None) -> None:
    X_to_use = X_val if X_val_override is None else X_val_override
    preds = fitted_estimator.predict(X_to_use)
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
stufe0_names = [name for name, _, _ in stufe0_specs]

fitted_baselines: dict = {}
_stufe0_timing: dict = {"family_durations": {}, "last_duration": None}
_log_progress(f"Starting Stufe 0: {len(stufe0_specs)} baselines.")
for i, (name, build_fn, fit_kwargs) in enumerate(stufe0_specs, start=1):
    fitted_baselines[name] = run_stage(
        name,
        build_fn,
        fit_kwargs,
        index=i,
        total=len(stufe0_specs),
        all_names=stufe0_names,
        timing_state=_stufe0_timing,
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
        # CatBoostClassifier has no clone()-compatible class_weights - see
        # build_catboost_pipeline's docstring. Weighting is applied via
        # sample_weight at fit time instead, exactly like xgboost_balanced
        # above (xgb_weights is CatBoost-compatible too: both use the same
        # "balanced" per-sample formula, so this is the same weighting
        # scheme as before, just applied via fit() instead of the
        # constructor).
        "catboost_balanced",
        lambda: build_catboost_pipeline(tree_preprocessor, use_gpu=_use_gpu_resolved),
        {"classify__sample_weight": xgb_weights},
    ),
]
stufe1_names = [name for name, _, _ in stufe1_specs]

fitted_models: dict = {}
_stufe1_timing: dict = {"family_durations": {}, "last_duration": None}
_log_progress(f"Starting Stufe 1: {len(stufe1_specs)} tree-ensemble configurations.")
for i, (name, build_fn, fit_kwargs) in enumerate(stufe1_specs, start=1):
    fitted_models[name] = run_stage(
        name,
        build_fn,
        fit_kwargs,
        index=i,
        total=len(stufe1_specs),
        all_names=stufe1_names,
        timing_state=_stufe1_timing,
    )
_log_progress(f"Stufe 1 complete: {len(stufe1_specs)}/{len(stufe1_specs)} (100%) models trained.")

# %% [markdown]
# ## 5 — Champion selection
#
# Selection is recall-gate-aware, not "highest macro-F1 wins alone."
# `random_forest_balanced` has the highest raw validation macro-F1 among the
# 8 Stufe 0/1 configurations (0.410), but its recall(class 1) is only 0.229 —
# far below the Q-phase acceptance gate (>= 0.50) — because that macro-F1
# edge comes from being conservative on the majority classes, exactly the
# wrong shape for this problem. `catboost_balanced` and `lightgbm_balanced`
# already clear the recall gate untuned, so **both** families advance as
# candidates to §6/§7 rather than a single macro-F1-only champion
# (baselines are never candidates — they exist to bound the floor, not to
# compete for it).

# %%
comparison_df = pd.DataFrame(comparison_rows)
tree_only = comparison_df[comparison_df["model"].isin(fitted_models.keys())]

# NOTE: candidate selection is recall-gate-aware, not "highest macro-F1
# wins alone". random_forest_balanced has the highest raw macro-F1 among
# the 8 Stufe 0/1 configurations (0.410) but recall(class 1)=0.229 - far
# below the Q-phase gate (>=0.50) - because RF's macro-F1 edge comes from
# being conservative on the majority classes, exactly the wrong shape for
# this problem. catboost_balanced (0.363 macro-F1, 0.537 recall) and
# lightgbm_balanced (0.364 macro-F1, 0.588 recall) already clear the
# recall gate untuned. docs/project/PROJEKTPLAN_SETUP.md's literature
# roadmap independently predicts Random Forest caps around 0.50-0.55
# macro-F1 while CatBoost/LightGBM have a 0.60-0.72 ceiling - so both
# families advance to §6/§7 as candidates; Random Forest/XGBoost/
# logistic/baselines stay in the table below for reporting only.
candidate_families = ["catboost", "lightgbm"]
candidate_names = {
    family: tree_only[tree_only["model"].str.startswith(family)]
    .sort_values("macro_f1", ascending=False)
    .iloc[0]["model"]
    for family in candidate_families
}
candidate_pipelines = {family: fitted_models[name] for family, name in candidate_names.items()}
for family, name in candidate_names.items():
    row = comparison_df[comparison_df["model"] == name].iloc[0]
    print(
        f"Candidate ({family}): {name}  macro-F1={row['macro_f1']:.3f}  "
        f"recall(1)={row['recall_class_1']:.3f}"
    )
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

# %% [markdown]
# ## 6 — Imbalance-strategy comparison (both candidate families)
#
# U-phase §10 menu, compared on each candidate family's base estimator, on a
# stratified subsample capped at 500,000 training rows (compute-budget soft
# constraint, Q-phase §9). Class weights are already reflected by whichever
# configuration won in §5 for each family — this section adds SMOTE, ADASYN,
# threshold moving, and ordinal classification on top of the *unweighted*
# variant of the same model family, so all five configurations are
# comparable on equal footing, for each of `catboost` and `lightgbm`
# independently.
#
# Reuses the same checkpoint-by-git-commit pattern from §0 (`_load_or_fit`
# below) so a crash partway through this section does not require redoing
# already-fitted strategies.


# %%
def _load_or_fit(name: str, fit_callable):
    """Generic checkpoint helper for Part 2 (arbitrary X/y per stage, unlike
    `_fit_or_checkpoint` in §0 which is hardcoded to X_train/y_train)."""
    path = CHECKPOINT_DIR / f"{name}.joblib"
    if path.exists():
        _log_progress(f"  -> {name}: loaded from checkpoint ({path.name})")
        return joblib.load(path)
    start = time.time()
    model = fit_callable()
    elapsed = time.time() - start
    joblib.dump(model, path)
    _log_progress(f"  -> {name} done in {elapsed:.1f}s")
    return model


SUBSAMPLE_CAP = 500_000
if len(X_train) > SUBSAMPLE_CAP:
    sample_idx = (
        y_train.groupby(y_train)
        .apply(lambda s: s.sample(frac=min(1.0, SUBSAMPLE_CAP / len(y_train)), random_state=42))
        .index.get_level_values(-1)
    )
    X_train_sub = X_train.loc[sample_idx].reset_index(drop=True)
    y_train_sub = y_train.loc[sample_idx].reset_index(drop=True)
    years_sub = train_df.loc[sample_idx, "UJAHR"].reset_index(drop=True)
else:
    X_train_sub, y_train_sub = X_train.reset_index(drop=True), y_train.reset_index(drop=True)
    years_sub = train_df["UJAHR"].reset_index(drop=True)
_log_progress(f"Imbalance-comparison subsample: {len(X_train_sub):,} rows (cap={SUBSAMPLE_CAP:,}).")


def _log_section6_progress(
    step_idx: int, total_steps: int, name: str, timing_state: dict, all_names: list[str]
) -> None:
    remaining = all_names[step_idx:]
    eta = _eta_seconds(remaining, timing_state["family_durations"], timing_state["last_duration"])
    eta_str = f"{eta / 60:.1f} min" if eta is not None else "unknown"
    pct = 100 * step_idx / total_steps
    _log_progress(
        f"[{step_idx + 1}/{total_steps}] ({pct:.0f}%) {name} ... ETA remaining: {eta_str}"
    )


def _log_section6_done(name: str, elapsed: float, timing_state: dict) -> None:
    timing_state["family_durations"].setdefault(name, []).append(elapsed)
    timing_state["last_duration"] = elapsed
    _log_progress(f"  -> {name} done in {elapsed:.1f}s")


# NOTE: unweighted_builder must return the *unweighted* variant for a fair
# comparison against SMOTE/ADASYN/ordinal (which are themselves the
# imbalance treatment) - CatBoost's own default (class_weights=None) and
# LightGBM need class_weight=None passed explicitly since its builder
# defaults to "balanced".
unweighted_builder = {
    "catboost": lambda pre, **kw: build_catboost_pipeline(pre, use_gpu=_use_gpu_resolved, **kw),
    "lightgbm": lambda pre, **kw: build_lightgbm_pipeline(
        pre, class_weight=None, use_gpu=_use_gpu_resolved, **kw
    ),
}

# SMOTE/ADASYN's k-NN search requires finite numeric input. IstGkfz is
# genuinely NaN for ~12.6% of rows (only recorded from 2018 onward, per
# docs/GLOSSARY.md) and tree_preprocessor's passthrough branch
# (scale_for_linear=False) deliberately leaves it untouched so tree models
# can use it as a native split signal - that's why Stufe 0/1 never
# crashed on it. SMOTE/ADASYN have no such native NaN handling and raise
# "ValueError: Input X contains NaN" if fed this output directly.
#
# A plain SimpleImputer is NOT sufficient here: ColumnTransformer's hstack
# of the passthrough bool/NaN column with the float columns produces an
# object-dtype array, and confirmed via direct inspection, IstGkfz's
# missing entries in that object array are Python `None`, not float
# `np.nan`. SimpleImputer's default NaN detection relies on
# self-inequality (`x != x`), which is only true for float NaN - `None`
# silently survives untouched, leaving residual NaN that crashes SMOTE/
# ADASYN downstream even after "imputing". Coercing through
# `pd.to_numeric(errors="coerce")` first canonicalizes every missing
# representation (None, np.nan, pd.NA, ...) to a proper np.nan, which
# SimpleImputer then correctly detects and fills.
#
# This preprocessing is family-independent (it only depends on
# tree_preprocessor and the subsample, not on which classifier follows
# it), so it is fit ONCE and shared by both families' SMOTE/ADASYN steps
# below - not refit per family.
fitted_preprocessor_for_resampling = clone(tree_preprocessor).fit(X_train_sub, y_train_sub)
_resampling_imputer = SimpleImputer(strategy="most_frequent")


def _numeric_impute(raw_transformed: np.ndarray, fit: bool) -> np.ndarray:
    numeric = pd.DataFrame(raw_transformed).apply(pd.to_numeric, errors="coerce")
    if fit:
        return _resampling_imputer.fit_transform(numeric)
    return _resampling_imputer.transform(numeric)


X_train_sub_transformed = _numeric_impute(
    fitted_preprocessor_for_resampling.transform(X_train_sub), fit=True
)
X_val_transformed_for_resampling = _numeric_impute(
    fitted_preprocessor_for_resampling.transform(X_val), fit=False
)

# %%
winning_strategy_per_family: dict[str, pd.Series] = {}

for family in candidate_families:
    build_unweighted = unweighted_builder[family]
    _section6_names = [
        f"{family}_smote",
        f"{family}_adasyn",
        f"{family}_threshold_moving",
        f"{family}_ordinal",
    ]
    _section6_timing: dict = {"family_durations": {}, "last_duration": None}

    def build_classifier_only():
        """Unfitted classify-step estimator only, for direct use on already-
        preprocessed numeric arrays (SMOTE/ADASYN output) - a full
        build_unweighted(tree_preprocessor) Pipeline would try to
        re-preprocess the already-transformed array as if it were the raw
        DataFrame."""
        return build_unweighted(tree_preprocessor).named_steps["classify"]

    # 6a: SMOTE
    _log_section6_progress(0, 4, f"{family}_smote", _section6_timing, _section6_names)
    _step_start = time.time()
    X_smote, y_smote = resample_smote(X_train_sub_transformed, y_train_sub)
    model_smote = _load_or_fit(
        f"{family}_smote", lambda: build_classifier_only().fit(X_smote, y_smote)
    )
    _score_on_validation(
        f"{family}_smote", model_smote, X_val_override=X_val_transformed_for_resampling
    )
    _log_section6_done(f"{family}_smote", time.time() - _step_start, _section6_timing)

    # 6b: ADASYN
    _log_section6_progress(1, 4, f"{family}_adasyn", _section6_timing, _section6_names)
    _step_start = time.time()
    X_adasyn, y_adasyn = resample_adasyn(X_train_sub_transformed, y_train_sub)
    model_adasyn = _load_or_fit(
        f"{family}_adasyn", lambda: build_classifier_only().fit(X_adasyn, y_adasyn)
    )
    _score_on_validation(
        f"{family}_adasyn", model_adasyn, X_val_override=X_val_transformed_for_resampling
    )
    _log_section6_done(f"{family}_adasyn", time.time() - _step_start, _section6_timing)

    # 6c: threshold moving (post-hoc on the unweighted family model)
    _log_section6_progress(2, 4, f"{family}_threshold_moving", _section6_timing, _section6_names)
    _step_start = time.time()
    model_unweighted = _load_or_fit(
        f"{family}_unweighted",
        lambda: build_unweighted(tree_preprocessor).fit(X_train_sub, y_train_sub),
    )
    val_proba = model_unweighted.predict_proba(X_val)
    best_threshold = find_best_threshold_for_class(
        y_val, val_proba, classes=list(model_unweighted.classes_), target_class=1
    )
    target_idx = list(model_unweighted.classes_).index(1)
    other_classes = np.array([c for c in model_unweighted.classes_ if c != 1])
    other_proba = np.delete(val_proba, target_idx, axis=1)
    fallback = other_classes[np.argmax(other_proba, axis=1)]
    threshold_preds = np.where(val_proba[:, target_idx] >= best_threshold, 1, fallback)
    threshold_metrics = evaluate_predictions(y_val, threshold_preds)
    comparison_rows.append({"model": f"{family}_threshold_moving", **threshold_metrics})
    print(f"[{family}] best threshold for class 1: {best_threshold:.2f}")
    _log_section6_done(f"{family}_threshold_moving", time.time() - _step_start, _section6_timing)

    # 6d: ordinal classification (Frank-Hall) using the family's own estimator as base
    _log_section6_progress(3, 4, f"{family}_ordinal", _section6_timing, _section6_names)
    _step_start = time.time()
    base_estimator_for_ordinal = clone(build_unweighted(tree_preprocessor).named_steps["classify"])
    ordinal_pipeline = _load_or_fit(
        f"{family}_ordinal",
        lambda: build_ordinal_pipeline(tree_preprocessor, base_estimator_for_ordinal).fit(
            X_train_sub, y_train_sub
        ),
    )
    _score_on_validation(f"{family}_ordinal", ordinal_pipeline)
    _log_section6_done(f"{family}_ordinal", time.time() - _step_start, _section6_timing)

    comparison_df = pd.DataFrame(comparison_rows)
    # Explicit allow-list, not a startswith(family) prefix match: the
    # latter also matches f"{family}_default" (a Stufe-1 baseline row with
    # no refit path in _build_pipeline_for below). Only the 4 strategies
    # actually compared in this loop, plus the family's own already-known
    # balanced candidate, are eligible to win. If a resampling/ordinal
    # strategy wins, _build_pipeline_for falls back to balanced with a
    # logged warning rather than crashing.
    _family_strategy_names = [
        f"{family}_smote",
        f"{family}_adasyn",
        f"{family}_threshold_moving",
        f"{family}_ordinal",
        candidate_names[family],
    ]
    strategy_rows = comparison_df[comparison_df["model"].isin(_family_strategy_names)]
    winning_row = select_best_candidate(strategy_rows)
    winning_strategy_per_family[family] = winning_row
    _log_progress(
        f"[{family}] winning strategy: {winning_row['model']}  "
        f"macro-F1={winning_row['macro_f1']:.3f}  recall(1)={winning_row['recall_class_1']:.3f}"
    )
    print(f"[{family}] winning strategy: {winning_row['model']}")

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.sort_values("macro_f1", ascending=False)


# CatBoostClassifier has no clone()-compatible class_weights (see
# build_catboost_pipeline's docstring for the full root cause) - both
# balanced_builder entries below build an UNWEIGHTED pipeline for
# CatBoost; its "balanced"-ness is applied via sample_weight at fit
# time instead (_fit_kwargs_for, below), exactly like xgboost_balanced
# in Stufe 1. LightGBM has no such issue - class_weight="balanced" is a
# plain string, which clones fine - so it stays a constructor kwarg.
balanced_builder = {
    "catboost": lambda pre, **kw: build_catboost_pipeline(pre, use_gpu=_use_gpu_resolved, **kw),
    "lightgbm": lambda pre, **kw: build_lightgbm_pipeline(
        pre, class_weight="balanced", use_gpu=_use_gpu_resolved, **kw
    ),
}


def _build_pipeline_for(family: str, strategy_model_name: str):
    """Unfitted pipeline matching the given (family, strategy) combination.

    SMOTE/ADASYN/threshold-moving/ordinal have no Optuna-compatible per-fold
    CV path (each needs its own resampling/thresholding step inside the fold,
    not just Pipeline.set_params().fit()). If one of them won §6, we fall back
    to the balanced candidate and log a warning — the §6 result is still
    recorded in the model card so the gap is visible, but the notebook doesn't
    crash and §7/§8 tune the best *implementable* strategy.
    """
    if strategy_model_name == candidate_names[family]:
        return balanced_builder[family](tree_preprocessor)
    if strategy_model_name == f"{family}_unweighted":
        return unweighted_builder[family](tree_preprocessor)
    _log_progress(
        f"[{family}] WARNING: winning strategy '{strategy_model_name}' has no §7 tuning path "
        f"(SMOTE/ADASYN/threshold-moving/ordinal require per-fold resampling inside Optuna CV). "
        f"Falling back to '{candidate_names[family]}' for tuning/refit."
    )
    return balanced_builder[family](tree_preprocessor)


def _fit_kwargs_for(family: str, strategy_model_name: str, y) -> dict:
    """Extra .fit()-time kwargs needed for the given (family, strategy)
    combination, for the given target array `y` (must match whatever X
    the returned kwargs will be used to fit - e.g. y_train_sub for the §7
    CV objective, y_train for the §8 full-data refit).

    Currently only CatBoost's balanced variant needs this: its
    class-weighting is applied via sample_weight at fit time rather than
    a constructor kwarg (see build_catboost_pipeline's docstring for why).
    LightGBM's balanced variant already bakes class_weight="balanced" into
    the pipeline in balanced_builder above, so it needs no extra fit
    kwargs here.
    """
    if family == "catboost" and strategy_model_name == candidate_names[family]:
        return {"classify__sample_weight": balanced_sample_weight(y)}
    return {}


# %% [markdown]
# ## 7 — Hyperparameter tuning (Optuna, per candidate family)
#
# Each candidate family's winning (model, strategy) combination from §6 is
# tuned **separately**: 9 trials for `catboost` and 9 trials for `lightgbm`
# (18 total — the same overall budget as tuning a single family at 40/18
# trials would have used, just split evenly across both surviving
# candidates instead of collapsing them into one champion before tuning).
#
# Each trial's mean CV macro-F1 remains the TPE sampler's single
# optimisation objective (unchanged direction), but every trial's mean CV
# recall(class 1) is also recorded via `trial.set_user_attr` — so, once a
# family's study finishes, `select_best_candidate` (not Optuna's own
# `study.best_trial`, which only ever tracked macro-F1) picks that family's
# best trial the same recall-gate-aware way §5/§6 already pick between
# configurations. This avoids the two studies drifting to hyperparameters
# that trade away recall(1) for a marginally higher macro-F1.
#
# **GPU reminder.** `objective()` below calls `_build_pipeline_for(family, ...)`
# once per trial per fold (`n_trials_per_family x n_groups_sub` fits total,
# per family) — by far the most fit-heavy loop in this notebook.
# `_build_pipeline_for` (§6) already threads `_use_gpu_resolved` into every
# LightGBM/CatBoost call, so this loop runs on GPU automatically whenever
# `USE_GPU` resolves to `True`.
#
# Each family's study is persisted to its own `study_name` inside the same
# commit-scoped SQLite file (`CHECKPOINT_DIR / "optuna_study.db"`), so a
# crash mid-tuning resumes each family from its own last completed trial
# instead of restarting either family's 9 trials — and so the two families'
# differing search spaces never collide inside one shared, resumed study.

# %%
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS_PER_FAMILY = 9  # 2 families x 9 = 18 total, same budget as the single-family run

# Both spaces include at least one regularisation-relevant parameter
# (l2_leaf_reg / reg_lambda, min_child_samples) in addition to capacity
# parameters (depth, leaves, estimators) - tuning capacity alone without
# any regularisation knob risks the search drifting toward overfit
# configurations in pursuit of subsample CV macro-F1.
PARAM_SPACES = {
    "catboost": lambda trial: {
        "classify__iterations": trial.suggest_int("iterations", 100, 500),
        "classify__depth": trial.suggest_int("depth", 3, 10),
        "classify__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "classify__l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
    },
    "lightgbm": lambda trial: {
        "classify__n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "classify__num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "classify__max_depth": trial.suggest_int("max_depth", 3, 12),
        "classify__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "classify__min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "classify__reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
    },
}

n_groups_sub = min(5, years_sub.nunique())
print(
    f"Optuna CV folds (n_groups_sub): {n_groups_sub}; distinct years in subsample: {years_sub.nunique()}"
)

_recall1_scorer = make_scorer(
    lambda y_true, y_pred: recall_for_class(y_true, y_pred, target_class=1)
)

tuned_candidates: dict[str, dict] = {}

for family in candidate_families:
    winning_strategy_name = winning_strategy_per_family[family]["model"]

    # NOTE (root-caused via systematic debugging after two failed patches):
    # sklearn's OWN cross_validate()/cross_val_score() clone the estimator
    # internally once per fold - unconditionally, regardless of how this
    # pipeline is built or passed in. Two earlier attempts to route around
    # that (rebuilding fresh instead of cloning a fitted model; removing
    # our own clone() call from this function) both failed identically,
    # because neither addressed the real incompatibility: CatBoostClassifier
    # configured with a non-None class_weights (list or dict) can NEVER
    # survive ANY clone() call, caller-side or internal to sklearn, since
    # CatBoost's own __init__/get_params() does not preserve that
    # parameter's object identity. The actual fix was upstream, in
    # build_catboost_pipeline() itself: class_weights was removed from the
    # constructor entirely, so every pipeline this function can return now
    # clones cleanly no matter who calls clone() on it. The "balanced"
    # class-weighting CatBoost needs is supplied separately via
    # _fit_kwargs_for()'s sample_weight, passed through cross_validate's
    # own params= argument below (fold-safe: sklearn slices sample_weight
    # to match each fold's training indices automatically).
    def objective(
        trial: optuna.Trial, family=family, winning_strategy_name=winning_strategy_name
    ) -> float:
        params = PARAM_SPACES[family](trial)
        pipeline = _build_pipeline_for(family, winning_strategy_name).set_params(**params)
        cv_results = cross_validate(
            pipeline,
            X_train_sub,
            y_train_sub,
            cv=GroupKFold(n_splits=n_groups_sub),
            groups=years_sub,
            scoring={"macro_f1": "f1_macro", "recall_1": _recall1_scorer},
            params=_fit_kwargs_for(family, winning_strategy_name, y_train_sub),
            n_jobs=1,
        )
        trial.set_user_attr("recall_class_1", float(cv_results["test_recall_1"].mean()))
        return float(cv_results["test_macro_f1"].mean())

    _trial_durations: list[float] = []

    def _progress_callback(
        study: "optuna.Study", trial: "optuna.trial.FrozenTrial", family=family
    ) -> None:
        elapsed = trial.duration.total_seconds() if trial.duration else 0.0
        _trial_durations.append(elapsed)
        avg = sum(_trial_durations) / len(_trial_durations)
        remaining = N_TRIALS_PER_FAMILY - (trial.number + 1)
        eta_min = (avg * remaining) / 60
        _log_progress(
            f"[Optuna {family} {trial.number + 1}/{N_TRIALS_PER_FAMILY}] "
            f"trial macro-F1={trial.value:.3f} recall(1)={trial.user_attrs['recall_class_1']:.3f} "
            f"in {elapsed:.1f}s ... ETA remaining: {eta_min:.1f} min"
        )

    optuna_db_path = CHECKPOINT_DIR / "optuna_study.db"
    study = optuna.create_study(
        study_name=f"a3_tuning_{family}",
        storage=f"sqlite:///{optuna_db_path}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    remaining_trials = max(0, N_TRIALS_PER_FAMILY - len(study.trials))
    _log_progress(
        f"[{family}] Optuna study: {len(study.trials)}/{N_TRIALS_PER_FAMILY} trials already "
        f"completed (persisted at {optuna_db_path.name}, study_name=a3_tuning_{family}); "
        f"{remaining_trials} remaining."
    )
    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, callbacks=[_progress_callback])

    trials_df = pd.DataFrame(
        [
            {
                "params": t.params,
                "macro_f1": t.value,
                "recall_class_1": t.user_attrs["recall_class_1"],
            }
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
    )
    best_trial_row = select_best_candidate(trials_df)
    tuned_candidates[family] = {
        "best_params": best_trial_row["params"],
        "cv_macro_f1": float(best_trial_row["macro_f1"]),
        "cv_recall_class_1": float(best_trial_row["recall_class_1"]),
    }
    _log_progress(
        f"[{family}] best tuned trial (gate-aware selection): "
        f"macro-F1={tuned_candidates[family]['cv_macro_f1']:.3f} "
        f"recall(1)={tuned_candidates[family]['cv_recall_class_1']:.3f} "
        f"params={tuned_candidates[family]['best_params']}"
    )
    print(f"[{family}] best tuned params: {tuned_candidates[family]['best_params']}")

# %% [markdown]
# ## 8 — Refit on full training data and evaluate on test-2024 exactly once
#
# The tuned configuration is refit on the **full** 2016–2022 training set
# (not the subsample used for tuning), then evaluated on the 2024 test
# split — the single time this notebook touches the test set.

# %%
# Pick the final cross-family winner using each family's TUNED CV scores
# (not a second full-data validation refit for both candidates - that
# would roughly double the expensive full-refit cost just to compare two
# candidates, for a comparison that CV, using year-grouped folds, already
# answers with acceptable noise for this purpose).
family_comparison = pd.DataFrame(
    [
        {"model": family, "macro_f1": v["cv_macro_f1"], "recall_class_1": v["cv_recall_class_1"]}
        for family, v in tuned_candidates.items()
    ]
)
final_row = select_best_candidate(family_comparison)
final_family = final_row["model"]
_log_progress(
    f"Final champion (post-tuning, gate-aware selection across families): {final_family}  "
    f"CV macro-F1={final_row['macro_f1']:.3f}  CV recall(1)={final_row['recall_class_1']:.3f}"
)
print(f"Final champion: {final_family}")
family_comparison.sort_values("macro_f1", ascending=False)

# %%
_log_progress(
    f"Refitting {final_family}'s tuned configuration on the FULL 2016-2022 training set..."
)
final_winning_strategy_name = winning_strategy_per_family[final_family]["model"]
final_best_params = {
    f"classify__{k}": v for k, v in tuned_candidates[final_family]["best_params"].items()
}

final_pipeline = _load_or_fit(
    f"{final_family}_final_tuned",
    lambda: (
        _build_pipeline_for(final_family, final_winning_strategy_name)
        .set_params(**final_best_params)
        .fit(
            X_train,
            y_train,
            **_fit_kwargs_for(final_family, final_winning_strategy_name, y_train),
        )
    ),
)

test_preds = final_pipeline.predict(X_test)
final_metrics = evaluate_predictions(y_test, test_preds)
passes = meets_acceptance_criteria(final_metrics)

_log_progress(
    f"FINAL TEST-2024 ({final_family}): macro-F1={final_metrics['macro_f1']:.3f} "
    f"recall(1)={final_metrics['recall_class_1']:.3f} gate={'PASS' if passes else 'FAIL'}"
)
print("=== FINAL TEST-2024 EVALUATION ===")
print(f"Champion family: {final_family}")
print(f"macro-F1:        {final_metrics['macro_f1']:.3f}  (threshold >= 0.55)")
print(f"recall(class 1): {final_metrics['recall_class_1']:.3f}  (threshold >= 0.50)")
print(f"Q-phase acceptance gate: {'PASS' if passes else 'FAIL'}")
print("Confusion matrix (rows=true, cols=pred, order=[1,2,3]):")
print(np.array(final_metrics["confusion_matrix"]))

# %% [markdown]
# ## 9 — Save the winning pipeline and model card

# %%
model_path = PROCESSED_DIR / "a3_best_model.joblib"
joblib.dump(final_pipeline, model_path)

model_card = {
    "champion_family": final_family,
    "winning_strategy": final_winning_strategy_name,
    "candidate_families_considered": candidate_families,
    "selection_rule": "highest macro-F1 among candidates with recall(class_1) >= 0.50; "
    "falls back to highest (macro_f1 + recall_class_1) / 2 if none clear the gate",
    "per_family_untuned_comparison": {
        family: winning_strategy_per_family[family].to_dict() for family in candidate_families
    },
    "per_family_tuned_cv_scores": tuned_candidates,
    "tuned_hyperparameters": tuned_candidates[final_family]["best_params"],
    "test_2024_metrics": final_metrics,
    "acceptance_gate_passed": passes,
    "provenance": provenance,
}
card_path = PROCESSED_DIR / "a3_model_card.json"
card_path.write_text(json.dumps(model_card, indent=2, default=str))
model_size_mb = model_path.stat().st_size / 1_048_576
_log_progress(f"Saved {model_path.name} ({model_size_mb:.1f} MB) and {card_path.name}.")
print(f"Saved: {model_path} ({model_size_mb:.1f} MB)")
print(f"Saved: {card_path}")

comparison_csv_path = PROCESSED_DIR / "a3_model_comparison.csv"
comparison_df.sort_values("macro_f1", ascending=False).to_csv(comparison_csv_path, index=False)
_log_progress(
    f"Saved full model-comparison table -> {comparison_csv_path.name} ({len(comparison_df)} rows)."
)
print(f"Saved: {comparison_csv_path}")

# %% [markdown]
# ## 10 — A³ summary and A³-to-C handoff
#
# ### Model-comparison table (central portfolio artefact)
#
# See §5 and §6 above — 8 baseline/tree configurations + up to 8 imbalance
# strategies (4 per candidate family: CatBoost, LightGBM) + 2 tuned final
# configurations, all scored on the 2023 validation split (Optuna tuning
# additionally cross-validated on the training subsample), with exactly
# one test-2024 evaluation for the final cross-family winner.
#
# ### A³-phase acceptance checklist
#
# ```text
# [ ] U-phase §10 preprocessing implemented as a single sklearn Pipeline
# [ ] Baselines (random guess, majority class, logistic regression) scored
# [ ] 4 tree families x {default, class-weighted} scored on validation
# [ ] Champion candidates (CatBoost, LightGBM) selected by a recall-gate-
#     aware rule, not macro-F1 alone (Random Forest's macro-F1 "win" had
#     recall(class 1)=0.229, far below the Q-phase gate)
# [ ] SMOTE / ADASYN / threshold moving / ordinal classification compared
#     on both candidate families
# [ ] Each family's winning (model, strategy) combination tuned with
#     Optuna (<= 9 trials per family, 18 total)
# [ ] Final cross-family champion selected by the same recall-gate-aware
#     rule, then refit on the FULL 2016-2022 training set
# [ ] Exactly one evaluation on the 2024 test set
# [ ] PASS/FAIL stated explicitly against macro-F1 >= 0.55 and recall(1) >= 0.50
# [ ] Winning pipeline + model card saved to data/processed/
# [ ] No SHAP, no literature discussion, no Streamlit work in this notebook
# ```
#
# > **Transition.** The winning pipeline and its model card are saved to
# > `data/processed/`. Proceed to `04_C_Phase.ipynb` for SHAP-based
# > explainability, benchmark comparison against the literature anchor
# > (Q-phase §11), and the limitations discussion.
