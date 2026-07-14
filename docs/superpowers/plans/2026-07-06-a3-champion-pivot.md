# A³ Champion Pivot — Recall-Aware Selection (CatBoost/LightGBM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the unfinished A³-phase work (`docs/superpowers/plans/2026-07-01-a3-phase-modelling.md`, Task 10/11) by replacing its "champion = highest validation macro-F1" selection rule with a recall-aware rule, carrying **CatBoost and LightGBM** forward as the two candidates (not Random Forest), tuning and evaluating them, and completing the still-pending documentation task — while touching the 2024 test set exactly once.

**Why this supersedes Task 10's §5 rule:** the original plan's step 4 ("Selecting a single champion = the (model, class-weight-setting) combination with the highest validation macro-F1") produced `random_forest_balanced` (macro-F1 0.410) — but that model's recall(class 1) is 0.229, far below the Q-phase's own ≥0.50 gate, while `catboost_balanced` (0.363 macro-F1, 0.537 recall) and `lightgbm_balanced` (0.364 macro-F1, 0.588 recall) already clear the harder gate *untuned*. `docs/project/PROJEKTPLAN_SETUP.md`'s literature-derived roadmap independently corroborates this: it predicts Random Forest caps around 0.50-0.55 macro-F1, while CatBoost+threshold-moving/Ordinal CatBoost is predicted to reach 0.65-0.72. Continuing to tune Random Forest is very unlikely to ever satisfy the recall gate; this plan pivots to the two families that already clear it.

**Architecture:** Notebook-orchestration changes only — no new files. `notebooks/03_A3_Phase.ipynb`'s §5-§10 are rewritten to run the existing §6 imbalance-strategy menu (SMOTE/ADASYN/threshold-moving/ordinal, already implemented in `src/unfallatlas/models/imbalance.py` and `ordinal.py`) over **two** candidate families instead of one, select within each family using a new shared helper, tune each family's winner separately with Optuna (still CV-only, never touching test), pick one final cross-family winner using the same helper, then refit once on full training data and evaluate once on test-2024. The new selection rule is added as a small, independently-testable pure function in `src/unfallatlas/models/evaluate.py` (matching the project's "reusable logic lives in `src/`, notebook orchestrates" convention) rather than inline notebook code, since it is called three times.

**Tech Stack:** Same as the existing A³ notebook — scikit-learn, imbalanced-learn, Optuna (TPE sampler), CatBoost, LightGBM, pandas, pytest. No new dependencies.

## Tooling for Implementers

This repo has `codebase-memory-mcp` (a code knowledge graph) and Serena (semantic code tools) available. Every task below touches either `src/unfallatlas/` (real Python modules) or `notebooks/03_A3_Phase.ipynb` (a Jupyter notebook) — use the right tool for each, not grep/cat, to save tokens and get more precise results:

- **`src/unfallatlas/models/evaluate.py` and `tests/test_evaluate.py` (Task 1):** use Serena's `get_symbols_overview` first, then `find_symbol(..., include_body=true)` to read `meets_acceptance_criteria`/`recall_for_class` before adding code near them, and `insert_after_symbol`/`replace_symbol_body` to add `select_best_candidate` — not a raw `Read`+`Edit` pair. Before writing the new function, confirm there's no existing similarly-named helper with `mcp__codebase-memory-mcp__search_graph(name_pattern=".*select.*|.*best.*candidate.*", project="home-jonas-Documents-Code-unfallatlas-qua3ck")` rather than grepping the tree.
- **Locating any function signature or call site referenced by this plan** (e.g. `build_catboost_pipeline`, `evaluate_predictions`, `find_best_threshold_for_class`, `build_ordinal_pipeline`, or checking nothing still calls the retired `build_champion`/`champion_builder` names after Task 3): use `mcp__codebase-memory-mcp__search_graph` (by `name_pattern` or `query`) and `mcp__codebase-memory-mcp__get_code_snippet(qualified_name=...)`, or Serena's `find_symbol`/`find_referencing_symbols` — both return exact signatures/docstrings/call sites in one call, instead of a grep-then-Read round trip. Use `mcp__codebase-memory-mcp__trace_path` if a task needs to understand a call chain (e.g. how `_load_or_fit` is used across §6/§7/§8).
- **`notebooks/03_A3_Phase.ipynb` (Tasks 2-5):** neither Serena nor codebase-memory-mcp parses notebook JSON as code (and `docs/superpowers/` /`.ipynb` content is outside the indexed repo scope for prose/plan lookups) — `NotebookEdit` is the correct and only tool for cell edits, per the project's notebook-source-of-truth convention. Do not fall back to hand-editing `notebooks/03_A3_Phase.py`.
- **After Task 1's commit changes `src/unfallatlas/`:** re-run `mcp__codebase-memory-mcp__index_repository(repo_path=..., mode="fast")` (fast mode is enough for a single-function addition) so `select_best_candidate` is discoverable via `search_graph`/`get_code_snippet` for the remaining tasks and any later review pass, instead of relying on stale graph state.
- **General architecture questions** (e.g. "what else in `src/unfallatlas/` reads `champion_pipeline`-shaped objects", "is there already a cross-validation helper") — use `mcp__codebase-memory-mcp__get_architecture` or `search_graph` with a natural-language `query` before assuming the answer or grepping.

## Global Constraints

Copied verbatim from `docs/superpowers/plans/2026-07-01-a3-phase-modelling.md`'s Global Constraints (still fully binding), plus one explicit amendment (marked ⚠️ AMENDED):

- **Target column:** `UKATGEORIE` ∈ {1=Getötet, 2=Schwer verletzt, 3=Leicht verletzt}, ordinal, class shares ≈ 1 % / 18 % / 81 %.
- **Primary metric:** macro-F1 on the held-out **2024** test year, acceptance threshold **≥ 0.55**.
- **Secondary metric:** recall for class 1 (Getötet) on the same test year, acceptance threshold **≥ 0.50**. A model failing either threshold does not meet the Q-phase acceptance gate — the notebook must state this explicitly, not omit it.
- **Split (fixed, chronological, never random):** train = `UJAHR <= 2022`, val = `UJAHR == 2023`, test = `UJAHR == 2024`.
- **Cross-validation inside the training window:** `GroupKFold` grouped by `UJAHR` — never a random `StratifiedKFold`.
- **Preprocessing is fully specified by U-phase §10** (`unfallatlas.features.preprocessing.build_preprocessor`) — this plan does not modify it or add engineered features.
- ⚠️ **AMENDED — champion selection.** Original rule: "highest validation macro-F1 among the 8 tree configurations wins, alone." New rule (this plan): **only `catboost_balanced` and `lightgbm_balanced` advance** past §5 (Random Forest/XGBoost/logistic/baselines stay in the reporting table but are not tuned further, since RF structurally fails the recall gate — see rationale above). Within and across those two families, the winner is chosen by `select_best_candidate()` (Task 1 below): highest macro-F1 **among candidates that clear recall(class 1) ≥ 0.50**; if none clear it, highest `(macro_f1 + recall_class_1) / 2`.
- **Compute budget (soft constraint, Q-phase §9):** single workstation. Optuna tuning and the SMOTE/ADASYN comparison run on the existing bounded, stratified 500,000-row subsample; the final chosen configuration is refit on the **full** 2016-2022 training set before the single test-2024 evaluation. Two families now share the §6/§7 budget: 9 Optuna trials per family (18 total, same as the current single-family budget) rather than 40.
- **Model size (soft constraint, Q-phase §9):** deployable artifact < 500 MB.
- **Interpretability (hard constraint, Q-phase §9):** the winning model must be SHAP-explainable (true of both CatBoost and LightGBM) — no SHAP analysis in this notebook (Phase C's job).
- **Reproducibility (hard constraint, Q-phase §9):** every run logs library versions, git commit, dataset hash, `random_state=42` (already implemented in `provenance` at notebook §0 — untouched by this plan).
- **Exactly one test-2024 evaluation, ever** — this is the single most important constraint for this plan specifically, since it now has two tuned candidates to choose between. The two-family comparison at every stage (§6, §7) uses **only** the 2023 validation split / CV on the training subsample — never `X_test`/`y_test`. `X_test`/`y_test` are referenced exactly once, in the rewritten §8.
- **Notebook policy:** `notebooks/*.ipynb` is source of truth; `.py` is a Jupytext percent-format mirror regenerated via `jupytext --sync`, **never hand-edited**. Use `NotebookEdit` on the `.ipynb`, then `uv run jupytext --sync notebooks/03_A3_Phase.ipynb` to regenerate the `.py` mirror.
- **Code conventions:** ruff + black formatting (pre-commit enforces this — `git commit` will auto-reformat and require a second `git add`+`commit` if the first attempt trips the hooks; this is expected, not a failure to fix by hand), line length 100, Python ≥3.11, no `print()` in `src/unfallatlas/` (notebooks may `print`).
- **Checkpointing:** `CHECKPOINT_DIR = data/processed/a3_checkpoints/<git-short-sha>/`, keyed by commit — a new commit gets a fresh (empty) directory, so still-valid checkpoints must be copied forward manually (`cp -n old/*.joblib new/`) after each commit in this plan, exactly as done in prior sessions. Never blindly `rm -rf` the whole checkpoint tree; only delete entries whose *config actually changed* (see per-task notes).
- **No race conditions in Optuna study naming.** Two families are now tuned in the same run — each MUST get its own `study_name` (`f"a3_tuning_{family}"`) so trials from one family's search space are never loaded into the other's study (this exact class of bug — a resumed study silently mixing an old, incompatible search space — was already hit once this session and is why last time's `optuna_study.db` had to be deleted rather than resumed).

---

## Task 1: `select_best_candidate()` — the shared, gate-aware selection rule

**Files:**
- Modify: `src/unfallatlas/models/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: nothing new — operates on the same `dict`/`pd.DataFrame` shape already produced by `evaluate_predictions()` (`macro_f1`, `recall_class_1`, ... keys).
- Produces: `select_best_candidate(rows: pd.DataFrame, recall_threshold: float = RECALL_CLASS_1_THRESHOLD) -> pd.Series` — used by Tasks 3, 4, and 5 in the notebook (three call sites: within-family strategy selection in §6, per-family best-Optuna-trial selection in §7, final cross-family selection before §8).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_evaluate.py`:

```python
import pandas as pd

from unfallatlas.models.evaluate import select_best_candidate


def test_select_best_candidate_picks_highest_macro_f1_among_recall_passers():
    rows = pd.DataFrame(
        [
            {"model": "a", "macro_f1": 0.50, "recall_class_1": 0.60},
            {"model": "b", "macro_f1": 0.60, "recall_class_1": 0.30},  # fails recall gate
            {"model": "c", "macro_f1": 0.45, "recall_class_1": 0.55},
        ]
    )
    winner = select_best_candidate(rows)
    assert winner["model"] == "a"  # highest macro_f1 among recall>=0.5 rows (a, c) is a


def test_select_best_candidate_falls_back_to_combined_score_if_none_pass():
    rows = pd.DataFrame(
        [
            {"model": "a", "macro_f1": 0.50, "recall_class_1": 0.10},  # combined 0.30
            {"model": "b", "macro_f1": 0.40, "recall_class_1": 0.45},  # combined 0.425
        ]
    )
    winner = select_best_candidate(rows)
    assert winner["model"] == "b"


def test_select_best_candidate_custom_threshold():
    rows = pd.DataFrame(
        [
            {"model": "a", "macro_f1": 0.50, "recall_class_1": 0.42},
            {"model": "b", "macro_f1": 0.45, "recall_class_1": 0.20},
        ]
    )
    winner = select_best_candidate(rows, recall_threshold=0.40)
    assert winner["model"] == "a"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluate.py -v -k select_best_candidate`
Expected: FAIL with `ImportError: cannot import name 'select_best_candidate'`

- [ ] **Step 3: Implement `select_best_candidate`**

Add to `src/unfallatlas/models/evaluate.py` (after `meets_acceptance_criteria`):

```python
import pandas as pd


def select_best_candidate(
    rows: pd.DataFrame, recall_threshold: float = RECALL_CLASS_1_THRESHOLD
) -> pd.Series:
    """Pick the best row from a (family, strategy) comparison table.

    Rule: highest ``macro_f1`` among rows clearing ``recall_class_1 >=
    recall_threshold`` (the harder Q-phase gate). If no row clears it,
    fall back to the highest ``(macro_f1 + recall_class_1) / 2`` combined
    score across all rows, so there is always a well-defined winner even
    when nothing meets the gate yet.

    This directly encodes "both acceptance criteria must pass" instead of
    optimising macro-F1 alone and hoping recall follows — the mistake that
    picked an unweighted-recall Random Forest as champion in the original
    A³ selection rule.
    """
    passing = rows[rows["recall_class_1"] >= recall_threshold]
    if len(passing) > 0:
        return passing.sort_values("macro_f1", ascending=False).iloc[0]
    combined = rows.assign(_combined_score=(rows["macro_f1"] + rows["recall_class_1"]) / 2)
    return combined.sort_values("_combined_score", ascending=False).iloc[0]
```

Add `import pandas as pd` to the top of the file if not already present (it is not — `evaluate.py` currently only imports from `sklearn.metrics`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: all tests pass, including the 3 new ones and the 5 pre-existing ones.

- [ ] **Step 5: Commit**

```bash
git add src/unfallatlas/models/evaluate.py tests/test_evaluate.py
git commit -m "feat: add select_best_candidate, a recall-aware champion selection rule"
```

---

## Task 2: Notebook §5 — carry forward two candidates instead of one champion

**Files:**
- Modify: `notebooks/03_A3_Phase.ipynb` (the champion-selection cell, currently `notebooks/03_A3_Phase.py:440-464`)

**Interfaces:**
- Consumes: `comparison_df`, `fitted_models` (dict keyed by e.g. `"catboost_balanced"`), `select_best_candidate` (Task 1), `FIG_DIR` — all already defined earlier in the notebook.
- Produces: `candidate_families: list[str] = ["catboost", "lightgbm"]`, `candidate_names: dict[str, str]` (e.g. `{"catboost": "catboost_balanced", "lightgbm": "lightgbm_balanced"}`), `candidate_pipelines: dict[str, Pipeline]` (fitted, from `fitted_models`) — all three consumed by Task 3.

- [ ] **Step 1: Replace the single-champion cell**

Using `NotebookEdit` on the cell whose current source starts with `comparison_df = pd.DataFrame(comparison_rows)` (right after the `## 5 — Champion selection` markdown cell), replace with:

```python
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
```

Leave the following `fig = px.bar(...)` cell (the Stufe 0+1 comparison plot) untouched — it still plots all 8+3 baseline/tree rows and remains a useful portfolio artefact.

- [ ] **Step 2: Sync jupytext and lint**

Run:
```bash
uv run jupytext --sync notebooks/03_A3_Phase.ipynb
uv run ruff check notebooks/03_A3_Phase.py
uv run python -c "import ast; ast.parse(open('notebooks/03_A3_Phase.py').read())"
```
Expected: ruff passes; `ast.parse` raises no exception (full-notebook execution happens once, at the end of Task 5 — don't re-execute after every task, since §0-§4 alone take real time to load/checkpoint-restore).

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_A3_Phase.ipynb notebooks/03_A3_Phase.py
git commit -m "feat: carry forward catboost+lightgbm as champion candidates, not RF"
```

---

## Task 3: Notebook §6 — imbalance-strategy comparison over both families

**Files:**
- Modify: `notebooks/03_A3_Phase.ipynb` (the entire §6 section, currently `notebooks/03_A3_Phase.py:473-685`)

**Interfaces:**
- Consumes: `candidate_families`, `candidate_names`, `candidate_pipelines` (Task 2); `select_best_candidate` (Task 1); `resample_smote`, `resample_adasyn`, `find_best_threshold_for_class` (`unfallatlas.models.imbalance`, unchanged); `build_ordinal_pipeline` (`unfallatlas.models.ordinal`, unchanged); `_load_or_fit`, `_log_section6_progress`, `_log_section6_done`, `_eta_seconds` (unchanged helpers already defined in this section).
- Produces: `winning_strategy_per_family: dict[str, pd.Series]` (one winning `(family, strategy)` row per family, e.g. `{"catboost": <row for catboost_threshold_moving>, "lightgbm": <row for lightgbm_balanced>}`) — consumed by Task 4.

This is the largest task in this plan. The existing single-family §6 code (SMOTE/ADASYN/threshold-moving/ordinal, `_numeric_impute` NaN fix for `IstGkfz`) is being generalized to loop over `candidate_families` instead of hardcoding one `champion_family`. The subsample construction, the NaN-safe preprocessing for resampling, and the four per-strategy steps are **structurally unchanged** — only the "which family" binding changes, from a single `champion_family` string to a loop variable.

- [ ] **Step 1: Add `select_best_candidate` to the §0 imports**

Using `NotebookEdit` on the §0 setup cell that currently reads
`from unfallatlas.models.evaluate import evaluate_predictions, meets_acceptance_criteria`
(`notebooks/03_A3_Phase.py:84`), change it to:

```python
from unfallatlas.models.evaluate import (
    evaluate_predictions,
    meets_acceptance_criteria,
    select_best_candidate,
)
```

- [ ] **Step 2: Replace the champion_builder + subsample-prep cell**

Using `NotebookEdit`, replace the cell currently starting with `def _load_or_fit(name: str, fit_callable):` through the cell ending at `def build_champion_classifier_only():` (i.e. everything from `notebooks/03_A3_Phase.py:490` to `:611`) with:

```python
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
```

- [ ] **Step 3: Replace the four per-strategy cells with a per-family loop**

Using `NotebookEdit`, replace the four cells `# 6a: SMOTE` through `# 6d: ordinal classification` (currently `notebooks/03_A3_Phase.py:614-678`) plus the winning-strategy-selection block that follows (`:680-709`, including the now-superseded `_build_winning_pipeline`) with:

```python
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
    _log_section6_done(
        f"{family}_threshold_moving", time.time() - _step_start, _section6_timing
    )

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
    strategy_rows = comparison_df[comparison_df["model"].str.startswith(family)]
    winning_row = select_best_candidate(strategy_rows)
    winning_strategy_per_family[family] = winning_row
    _log_progress(
        f"[{family}] winning strategy: {winning_row['model']}  "
        f"macro-F1={winning_row['macro_f1']:.3f}  recall(1)={winning_row['recall_class_1']:.3f}"
    )
    print(f"[{family}] winning strategy: {winning_row['model']}")

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.sort_values("macro_f1", ascending=False)


def _build_pipeline_for(family: str, strategy_model_name: str):
    """Unfitted pipeline matching the given (family, strategy) combination -
    NOT unweighted_builder's always-unweighted baseline, which exists only
    to give SMOTE/ADASYN/threshold-moving/ordinal a fair, equally-
    unweighted opponent in the §6 comparison above. Tuning/refitting the
    unweighted builder directly here would silently retune the wrong
    configuration regardless of which strategy actually won - e.g. if the
    plain balanced classifier (`candidate_pipelines[family]`) beat every
    resampling/ordinal treatment, §7/§8 must tune and refit *that*, not
    the unweighted variant.
    """
    if strategy_model_name == candidate_names[family]:
        return clone(candidate_pipelines[family])
    if strategy_model_name == f"{family}_unweighted":
        return unweighted_builder[family](tree_preprocessor)
    raise NotImplementedError(
        f"No tuning/refit path implemented for winning strategy '{strategy_model_name}' - "
        "SMOTE/ADASYN/threshold-moving/ordinal each need a fitting procedure "
        "inside Optuna's per-fold CV other than plain "
        "Pipeline.set_params().fit(), which is out of scope here."
    )
```

Note on the `NotImplementedError` branch: if SMOTE, ADASYN, or ordinal wins for a family in your actual run, Task 4's Optuna step for that family will raise this error when it calls `_build_pipeline_for`. This mirrors the limitation already accepted in the previous iteration of this notebook (documented in `.superpowers/sdd/progress.md`) — implementing per-fold-safe resampling inside `cross_val_score` is a real scope expansion (needs an `imblearn.pipeline.Pipeline` with the resampler as a fold-safe step) that this plan does not include. If this happens during Task 6's execution, stop and flag it to the user rather than silently working around it — do not disable the check.

- [ ] **Step 4: Sync jupytext and lint**

```bash
uv run jupytext --sync notebooks/03_A3_Phase.ipynb
uv run ruff check notebooks/03_A3_Phase.py
uv run python -c "import ast; ast.parse(open('notebooks/03_A3_Phase.py').read())"
```
Expected: ruff passes, no syntax errors. Also verify via Serena `search_graph`/`find_referencing_symbols` (or `search_code` as a fallback) that no reference to `champion_family`, `champion_builder`, `build_champion`, or `build_champion_classifier_only` remains anywhere below this cell in the notebook — these names are retired by this task.

- [ ] **Step 5: Delete checkpoints that must be retrained**

The four per-strategy checkpoints for the *old* single champion family no longer apply to the new candidate set. Random Forest's own §6 checkpoints (`random_forest_smote.joblib`, `random_forest_adasyn.joblib`, `random_forest_ordinal.joblib`, `random_forest_unweighted.joblib`) stay — they're unaffected, still valid, and still shown in the reporting table. Nothing needs deleting here yet: `catboost_smote`/`catboost_adasyn`/`catboost_threshold_moving`/`catboost_ordinal`/`catboost_unweighted` and the `lightgbm_*` equivalents don't exist yet, so there is nothing stale to remove — Task 6's execution will create them fresh.

- [ ] **Step 6: Commit**

```bash
git add notebooks/03_A3_Phase.ipynb notebooks/03_A3_Phase.py
git commit -m "feat: run imbalance-strategy comparison over both candidate families"
```

---

## Task 4: Notebook §7 — per-family Optuna tuning with recall tracked, not just macro-F1

**Files:**
- Modify: `notebooks/03_A3_Phase.ipynb` (the entire §7 section, currently `notebooks/03_A3_Phase.py:712-823`)

**Interfaces:**
- Consumes: `winning_strategy_per_family`, `_build_pipeline_for`, `candidate_families` (Task 3); `select_best_candidate` (Task 1); `recall_for_class` (`unfallatlas.models.evaluate`, already exists — used here for the first time in the notebook).
- Produces: `tuned_candidates: dict[str, dict]` — one entry per family with keys `"best_params"`, `"cv_macro_f1"`, `"cv_recall_class_1"` — consumed by Task 5.

**Why track recall during tuning, not just at selection time:** the original single-objective Optuna loop only ever optimised/reported macro-F1. Blindly taking `study.best_trial` (argmax macro-F1) risks picking hyperparameters that further *worsen* recall(1) in pursuit of macro-F1 — the exact same mistake as the old champion-selection rule, just one level deeper. This task keeps the TPE search itself single-objective (simpler, unchanged optimisation direction) but records each trial's recall(1) via `trial.set_user_attr`, then applies `select_best_candidate` over the completed trials to choose which trial's params to actually use — so a trial with slightly lower macro-F1 but a passing recall(1) can still win, consistent with the rest of this plan's selection rule.

- [ ] **Step 1: Replace the PARAM_SPACES + objective + study cell**

Using `NotebookEdit`, replace the cell from `optuna.logging.set_verbosity(...)` through the `print(f"Best params: ...")` line (currently `notebooks/03_A3_Phase.py:738-823`) with:

```python
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

_recall1_scorer = make_scorer(lambda y_true, y_pred: recall_for_class(y_true, y_pred, target_class=1))

tuned_candidates: dict[str, dict] = {}

for family in candidate_families:
    winning_strategy_name = winning_strategy_per_family[family]["model"]
    base_pipeline = _build_pipeline_for(family, winning_strategy_name)

    def objective(trial: optuna.Trial, family=family, base_pipeline=base_pipeline) -> float:
        params = PARAM_SPACES[family](trial)
        pipeline = clone(base_pipeline).set_params(**params)
        cv_results = cross_validate(
            pipeline,
            X_train_sub,
            y_train_sub,
            cv=GroupKFold(n_splits=n_groups_sub),
            groups=years_sub,
            scoring={"macro_f1": "f1_macro", "recall_1": _recall1_scorer},
            n_jobs=1,
        )
        trial.set_user_attr("recall_class_1", float(cv_results["test_recall_1"].mean()))
        return float(cv_results["test_macro_f1"].mean())

    _trial_durations: list[float] = []

    def _progress_callback(study: "optuna.Study", trial: "optuna.trial.FrozenTrial", family=family) -> None:
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
```

Add `from sklearn.metrics import make_scorer` and `from sklearn.model_selection import cross_validate` to the imports in the §0 setup cell (`notebooks/03_A3_Phase.py:56-91`) — `cross_val_score` (already imported) only supports a single scorer; `cross_validate` supports the `scoring={"macro_f1": ..., "recall_1": ...}` dict used above. Keep the existing `GroupKFold` import; `cross_val_score` itself is no longer called anywhere after this task and may be dropped from the import list. Also extend the `unfallatlas.models.evaluate` import (already touched in Task 3 Step 1) to include `recall_for_class`:

```python
from unfallatlas.models.evaluate import (
    evaluate_predictions,
    meets_acceptance_criteria,
    recall_for_class,
    select_best_candidate,
)
```

- [ ] **Step 2: Sync jupytext and lint**

```bash
uv run jupytext --sync notebooks/03_A3_Phase.ipynb
uv run ruff check notebooks/03_A3_Phase.py
uv run python -c "import ast; ast.parse(open('notebooks/03_A3_Phase.py').read())"
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_A3_Phase.ipynb notebooks/03_A3_Phase.py
git commit -m "feat: tune both candidate families separately, tracking recall(1) per trial"
```

---

## Task 5: Notebook §8 — final cross-family selection, single refit, single test evaluation

**Files:**
- Modify: `notebooks/03_A3_Phase.ipynb` (the entire §8 section, currently `notebooks/03_A3_Phase.py:825-860`)

**Interfaces:**
- Consumes: `tuned_candidates`, `candidate_families` (Task 4); `_build_pipeline_for` (Task 3); `select_best_candidate` (Task 1); `evaluate_predictions`, `meets_acceptance_criteria` (unchanged).
- Produces: `final_family: str`, `final_pipeline` (fitted), `final_metrics: dict`, `passes: bool` — consumed by Task 6 (model card).

This is the only place `X_test`/`y_test` may be referenced in the entire notebook.

- [ ] **Step 1: Replace the refit + evaluate cell**

Using `NotebookEdit`, replace the cell from `_log_progress("Refitting tuned configuration...")` through the confusion-matrix print (currently `notebooks/03_A3_Phase.py:833-860`) with:

```python
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
_log_progress(f"Refitting {final_family}'s tuned configuration on the FULL 2016-2022 training set...")
final_winning_strategy_name = winning_strategy_per_family[final_family]["model"]
final_best_params = {
    f"classify__{k}": v for k, v in tuned_candidates[final_family]["best_params"].items()
}

final_pipeline = _load_or_fit(
    f"{final_family}_final_tuned",
    lambda: (
        _build_pipeline_for(final_family, final_winning_strategy_name)
        .set_params(**final_best_params)
        .fit(X_train, y_train)
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
```

- [ ] **Step 2: Update §9's model card cell**

Using `NotebookEdit`, replace the `model_card = {...}` cell (currently `notebooks/03_A3_Phase.py:869-882`) with:

```python
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
```

(`default=str` added to `json.dumps` because `winning_strategy_per_family[family]` is a `pd.Series` whose `.to_dict()` values may include numpy scalar types that `json` cannot serialize directly.)

- [ ] **Step 3: Update §10's handoff markdown**

Using `NotebookEdit`, replace the markdown cell containing the acceptance checklist (currently `notebooks/03_A3_Phase.py:884-913`) — change the checklist items describing single-champion selection to reflect two candidates, and change "Champion selected by validation macro-F1" to "Champion candidates (CatBoost, LightGBM) selected by recall-gate-aware selection, not macro-F1 alone":

```markdown
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
```

- [ ] **Step 4: Sync jupytext and lint**

```bash
uv run jupytext --sync notebooks/03_A3_Phase.ipynb
uv run ruff check notebooks/03_A3_Phase.py
uv run python -c "import ast; ast.parse(open('notebooks/03_A3_Phase.py').read())"
```

- [ ] **Step 5: Commit**

```bash
git add notebooks/03_A3_Phase.ipynb notebooks/03_A3_Phase.py
git commit -m "feat: select final champion across both tuned families, single test-2024 eval"
```

---

## Task 6: Execute the full notebook end-to-end and verify

**Files:**
- Execute: `notebooks/03_A3_Phase.ipynb` (no code changes in this task — verification only)

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: `data/processed/a3_best_model.joblib`, `data/processed/a3_model_card.json`, updated `reports/a3_progress.log`, updated `data/processed/a3_checkpoints/<new-commit-sha>/`.

- [ ] **Step 1: Copy forward still-valid checkpoints to the new commit's directory**

After Task 5's commit, `CHECKPOINT_DIR` moves to a fresh, empty `data/processed/a3_checkpoints/<new-sha>/`. Copy forward everything that is still valid (Stufe 0/1 baselines, and Random Forest's already-computed §6 strategies, which remain correct and unaffected by this plan):

```bash
NEW_SHA=$(git rev-parse --short HEAD)
mkdir -p "data/processed/a3_checkpoints/$NEW_SHA"
cp -n data/processed/a3_checkpoints/56abd90/*.joblib "data/processed/a3_checkpoints/$NEW_SHA/"
ls "data/processed/a3_checkpoints/$NEW_SHA/"
```
(Adjust `56abd90` to whatever the actual prior checkpoint directory is named at execution time — check `ls data/processed/a3_checkpoints/` first.) Do NOT copy forward any `optuna_study.db` or `*_final_tuned.joblib` — those are tied to the old single-champion tuning logic and must be regenerated fresh under the new per-family study names.

- [ ] **Step 2: Run the notebook as a detached, separate process**

This run is long (18 Optuna trials across two families plus two full-data refits) — it must not die if the VSCode window, terminal tab, or Claude Code session closes/crashes. Tell the user to launch it detached from their current shell (`setsid` + `disown`, or `nohup`, either survives the launching shell exiting; `nohup` alone does not survive `SIGHUP` from a closed terminal unless also backgrounded and disowned):

```bash
setsid uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_A3_Phase.ipynb \
  > /tmp/a3_nbconvert.log 2>&1 < /dev/null &
disown
```

Confirm it's actually detached (still alive if you check a few seconds later, and not a child of the current shell's job table):
```bash
ps -ef | grep nbconvert | grep -v grep
```

Monitor progress via `tail -f reports/a3_progress.log` (this is safe to run/re-run anytime — it's a separate read-only process; do not launch the `nbconvert` command itself in the background from Claude Code's own Bash tool — the user runs and owns that process, per their established preference, precisely so it survives independently of this session).

Expected: exits 0, no exception cells. Watch for:
- `n_groups_sub` printed equal to the real distinct-year count in `years_sub` (confirms `GroupKFold` groups by real years, not an index-derived value).
- Both families' §6 comparison rows print with plausible macro-F1/recall(1) values (roughly in the 0.35-0.60 range given the untuned Stufe-1 numbers already observed).
- Optuna logs show two independent progress streams (`[Optuna catboost i/9]` and `[Optuna lightgbm i/9]`), each starting at `0/9 trials already completed` (confirms the per-family `study_name` split worked, not a resumed/mixed study).
- Exactly one `FINAL TEST-2024` log line.
- `data/processed/a3_best_model.joblib` exists and is under 500 MB (Q-phase §9 soft constraint) — if it is not, flag to the user before proceeding (do not silently accept an oversized artifact).

- [ ] **Step 3: Verify the acceptance-gate statement and model card**

Run: `uv run python -c "import json; print(json.dumps(json.load(open('data/processed/a3_model_card.json')), indent=2))"`
Expected: valid JSON; `acceptance_gate_passed` is explicitly `true` or `false` (never missing); `per_family_tuned_cv_scores` has exactly 2 entries (catboost, lightgbm); `champion_family` is one of `"catboost"`/`"lightgbm"`.

- [ ] **Step 4: Full test suite and lint**

Run: `uv run pytest -v && uv run ruff check . && uv run black --check .`
Expected: all tests pass (including Task 1's 3 new tests); ruff and black report no violations.

- [ ] **Step 5: Sync jupytext one final time and commit the executed notebook**

```bash
uv run jupytext --sync notebooks/03_A3_Phase.ipynb
git add notebooks/03_A3_Phase.ipynb notebooks/03_A3_Phase.py data/processed/a3_best_model.joblib data/processed/a3_model_card.json
git commit -m "feat: execute A3 champion pivot end-to-end, save final model + model card"
```

If `a3_best_model.joblib` exceeds the pre-commit hook's `check-added-large-files --maxkb=5120` limit, add `data/processed/a3_best_model.joblib` to `.gitattributes` for Git LFS tracking (matching the existing `data/accidents.parquet` pattern) before retrying the commit — do not bypass the hook with `--no-verify`.

---

## Task 7: Finish Task 11 of the original plan — documentation updates

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/GLOSSARY.md`

**Interfaces:**
- Consumes: `final_family`, `final_metrics`, `passes` (read from the executed notebook's printed output / `a3_model_card.json` — do not re-derive).

This is the original plan's still-pending Task 11, updated to reflect the new selection methodology instead of the old single-champion-by-macro-F1 wording.

- [ ] **Step 1: Update `AGENTS.md`'s notebook status table**

In `AGENTS.md`, within the `<!-- AUTO-MANAGED: architecture -->` block, change:
```
│   ├── 03_A3_Phase.ipynb   # Modelling & tuning (TODO)
```
to:
```
│   ├── 03_A3_Phase.ipynb   # Modelling & tuning (done)
```
and add the library files used by this phase to the same block's file listing, next to the existing `features/` and `models/` entries:
```
│   ├── features/           # enrich.py, spatial.py, temporal.py, preprocessing.py
│   ├── models/             # baseline.py, boosting.py, evaluate.py, ordinal.py, imbalance.py
```

- [ ] **Step 2: Add new terms to `docs/GLOSSARY.md`**

Append to the "## Machine Learning Concepts" section (after the existing `**sklearn Pipeline**` entry, before the `---` that precedes "## Process Model"):

```markdown
**Threshold Moving**
Adjusting the decision threshold for a specific class's predicted probability after training, instead of relying on the default arg-max rule. Used in A³ to raise recall on class 1 (Getötet) without retraining the model.

**Ordinal Classification (Frank–Hall decomposition)**
Decomposes a K-class ordinal target into K−1 binary "is y greater than threshold i" classifiers, then recovers per-class probabilities by differencing consecutive cumulative probabilities. Exploits the natural ordering of `UKATGEORIE` (1 < 2 < 3) rather than treating the classes as unordered categories.

**Optuna**
A hyperparameter-optimisation library using sequential model-based search (TPE — Tree-structured Parzen Estimator — by default). Used in A³ to tune the winning (model, imbalance-strategy) combination of each champion candidate, bounded to a fixed trial count per family to respect the project's single-workstation compute budget.

**GroupKFold**
A cross-validation splitter that guarantees all rows sharing a group value (here: `UJAHR`, the accident year) fall in the same fold. Used instead of a random `StratifiedKFold` to prevent a model from training and validating on the same year, which would leak temporal structure during model selection.

**Champion candidates**
The (model, configuration) combinations carried forward from Stufe 0/1 into the imbalance-strategy comparison and hyperparameter tuning. Selected by `select_best_candidate()`: highest validation macro-F1 among candidates that clear the recall(class 1) ≥ 0.50 gate, not by macro-F1 alone — Random Forest had the single highest raw macro-F1 among the 8 Stufe-1 configurations but recall(class 1) far below the gate, so CatBoost and LightGBM (which cleared the gate untuned) were carried forward instead.
```

- [ ] **Step 3: Run the full test suite once more and lint**

Run: `uv run pytest -v && uv run ruff check . && uv run black --check .`
Expected: all tests pass; ruff and black report no violations.

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md docs/GLOSSARY.md
git commit -m "docs: mark A3 phase complete, document recall-gate-aware champion selection"
```

---

## Self-Review Notes

- **Spec coverage:** Q-phase acceptance criteria (macro-F1 ≥0.55 AND recall(1) ≥0.50, both required) → Task 1 (`select_best_candidate`) + Task 5 (§8 gate statement); U-phase §10 CV hint (GroupKFold by year, never random) → unchanged, reused as-is in Task 4; U-phase §10 imbalance menu (class weights/SMOTE/ADASYN/threshold moving/ordinal) → Task 3, now applied to both candidate families; compute budget (≤500k subsample, single test touch) → Task 3/4/5 explicitly preserve both; model-size/interpretability/reproducibility constraints → untouched, inherited from the original plan; still-pending doc updates (original Task 11) → Task 7.
- **Placeholder scan:** no "TBD"/"handle edge cases" left; every code block is complete, runnable notebook cell content, not a description of what to write.
- **Type consistency:** `select_best_candidate(rows: pd.DataFrame, recall_threshold: float) -> pd.Series` used identically across Tasks 3 (strategy_rows), 4 (trials_df), and 5 (family_comparison) — all three call sites pass a `DataFrame` with `macro_f1`/`recall_class_1` columns, matching the Task 1 signature. `_build_pipeline_for(family: str, strategy_model_name: str)` (Task 3) is called identically in Task 4 (`base_pipeline = _build_pipeline_for(family, winning_strategy_name)`) and Task 5 (`_build_pipeline_for(final_family, final_winning_strategy_name)`). `tuned_candidates[family]` dict keys (`best_params`, `cv_macro_f1`, `cv_recall_class_1`) are produced in Task 4 and consumed with the same three keys in Task 5.
- **Race-condition/misconfiguration check:** each family gets its own Optuna `study_name` (`a3_tuning_catboost`, `a3_tuning_lightgbm`) in the same SQLite file — Optuna's storage backend supports multiple named studies per file natively, so there is no file-level contention and no risk of one family's trials being loaded into the other's search space (the exact bug class that forced deleting `optuna_study.db` earlier this session). The notebook itself is single-process/single-threaded at the orchestration level (loops run sequentially, not via multiprocessing), so no true concurrent-write race exists within a run; `n_jobs=1` is kept on `cross_validate` deliberately (matching the existing convention) to avoid nested-parallelism blowup, since the underlying estimators already parallelize internally (`n_jobs=-1` on Random Forest/LightGBM; GPU device on CatBoost/LightGBM/XGBoost when available).
- **Overfitting safeguards, explicit:** CatBoost tuning now includes `l2_leaf_reg` (previously untuned, defaulting to library default 3.0); LightGBM tuning now includes `max_depth` (previously unbounded, `-1` default) and `min_child_samples`/`reg_lambda` (previously fixed at build time, not searched) — capacity parameters (depth, leaves, iterations) are never tuned without at least one accompanying regularisation parameter in the same search space, consistent with the project's existing bounded-depth convention for Random Forest/XGBoost/LightGBM (`src/unfallatlas/models/boosting.py`).
