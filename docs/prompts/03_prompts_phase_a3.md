# AI Prompt for Phase A³

## Build the A³-Phase from the U-Phase Contract

**Claude Code (Sonnet 5), effort: medium (Anthropic, 2026):**

### Initial prompt

The A³-phase plan was produced with the `superpowers:writing-plans` skill
(Claude Code / superpowers plugin), starting from the following two
verbatim user messages:

> Now please do me a favour now the A Phase writing creation and
> implementation is the next step and i want you to first read all the
> @docs/ and especiall the relevant ones additionally the already existing
> Q and U Phase as it builds upon these 2 then craft a veeery detailed plan
> using /writing-plans where you make AN EXACT SCOPE what the exact things
> are that HAVE to be done in A Phase and what explicitly DONT HAS to go in
> the A PHase as i had issues previously about exactly this in Q and U
> phase with you so make this extra good. Then utilize all the information
> to make a extremly strong A Phase tailored perfectly to my project and
> its context. Then display the plan to me so i can approve it

> IMPORTANT write the finished approved plan as first task inside
> @docs/prompts/ folder as a new prompt file this one for A phase and
> update @"docs/AI TOOL DISCLOSURE.md"

This is the resulting Plan Content that was implemented using the `superpowers:subagent-driven-development` skill (Claude Code / superpowers plugin):

````markdown
# A³ Phase (Algorithm / Adapt / Adjust) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the A³ phase of the QUA³CK process for `unfallatlas-qua3ck` — a single notebook (`notebooks/03_A3_Phase.ipynb`) plus its supporting `src/unfallatlas/` library code — that takes the U-phase's §10 preprocessing contract, trains and compares baseline/tree/imbalance-aware models under a chronological split, tunes the winning configuration, and reports a single held-out 2024 test result against the Q-phase acceptance criteria.

**Architecture:** Reusable logic (preprocessing pipeline, model builders, metrics, imbalance strategies) lives in small, independently testable modules under `src/unfallatlas/{features,models}/`, each with a pytest suite. The notebook imports these modules and contains only orchestration, narrative, and visualisation — matching the project's existing "notebook imports library" convention (`AGENTS.md`: *"Reusable logic moves from notebook cells into `src/unfallatlas/` and is imported back into the notebook"*).

**Tech Stack:** pandas, DuckDB, scikit-learn ≥1.4, imbalanced-learn ≥0.12 (SMOTE/ADASYN), XGBoost ≥2, LightGBM ≥4.3, CatBoost ≥1.2, Optuna ≥3.6, Plotly ≥5.22, pytest, jupytext (percent format, paired with `.ipynb`). **No new dependencies are added to `pyproject.toml`** — every library used above is already declared.

## Global Constraints

These are copied verbatim from the Q-phase (`notebooks/01_Q_Phase.py`) and U-phase (`notebooks/02_U_Phase.py`) contracts. Every task below implements against them; no task may contradict them.

- **Target column:** `UKATGEORIE` ∈ {1=Getötet, 2=Schwer verletzt, 3=Leicht verletzt}, ordinal, class shares ≈ 1 % / 18 % / 81 %.
- **Primary metric:** macro-F1 on the held-out **2024** test year, acceptance threshold **≥ 0.55**.
- **Secondary metric:** recall for class 1 (Getötet) on the same test year, acceptance threshold **≥ 0.50**. A model failing either threshold does not meet the Q-phase acceptance gate — the notebook must state this explicitly, not omit it.
- **Split (fixed, chronological, never random):** train = `UJAHR <= 2022`, val = `UJAHR == 2023`, test = `UJAHR == 2024`.
- **Cross-validation inside the training window:** `GroupKFold` grouped by `UJAHR`, or `TimeSeriesSplit` — **never** a random `StratifiedKFold` (U-phase §10, "Cross-validation hint").
- **Preprocessing is fully specified by U-phase §10 — A³ implements it, it does not re-decide it:**
  - Drop before fit: `OBJECTID`, `UGEMEINDE`. `UJAHR` is retained only for splitting/CV grouping, then dropped before `.fit()`.
  - Cyclic sin/cos encoding: `UMONAT` (period 12), `USTUNDE` (period 24), `UWOCHENTAG` (period 7).
  - One-hot: `UART`, `UTYP1`, `ULICHTVERH`, `STRZUSTAND` (U-phase §9.1 leakage probe already cleared `UART`/`UTYP1` — entropy reduction 3.4 % / 2.2 %, both far below the 50 % trigger).
  - Target-mean encoding with additive smoothing: `UREGBEZ`, `UKREIS`.
  - Passthrough (no encoding needed): `IstRad`, `IstPKW`, `IstFuss`, `IstKrad`, `IstGkfz`, `IstSonstig`, `LON`, `LAT` — scaled only when the estimator is distance/linear-based (Logistic Regression), not for tree models.
  - `log1p` + median-0 imputation + `StandardScaler`: `dwd_precip_mm`, `dwd_visibility_m`.
  - Median imputation + `StandardScaler` (no transform): `dwd_temp_air_2m`, `dwd_wind_speed_ms`.
  - `log` transform (no scaling, no missing values by construction): `dwd_station_dist_km`.
  - All `fit`-based preprocessing statistics (imputation medians, target-encoding means, scaler parameters) are fit on the training fold only, via a single `sklearn.pipeline.Pipeline` — never on val/test (U-phase §10 "handover").
- **Imbalance handling — A³ chooses from the U-phase §10 menu, and this plan fixes the exact scope** (see "A³ Scope" section below) to keep compute bounded on ~2.09 M rows: class weights first (cheap, applied to all four tree families), then SMOTE / ADASYN / threshold moving / ordinal classification compared only on the single best-performing model from the first pass (the "champion").
- **Compute budget (soft constraint, Q-phase §9):** single workstation, no cluster. Optuna tuning and the SMOTE/ADASYN comparison run on a bounded, documented stratified subsample of the training set (cap: 500,000 rows) — the final chosen configuration is refit on the **full** 2016–2022 training set before the single test-2024 evaluation.
- **Model size (soft constraint, Q-phase §9):** deployable artifact < 500 MB (Streamlit runs on a home server).
- **Interpretability (hard constraint, Q-phase §9):** the winning model must be SHAP-explainable — this rules out nothing in scope here (all candidate models support SHAP), but SHAP analysis itself is **out of scope for A³** (see below).
- **Reproducibility (hard constraint, Q-phase §9):** every run logs library versions, git commit, dataset hash, and a fixed `random_state=42` — same provenance-block pattern as `notebooks/02_U_Phase.py` §0.
- **Notebook policy (`AGENTS.md`):** `notebooks/*.ipynb` are source of truth; `.py` files are Jupytext percent-format mirrors regenerated via `jupytext --sync`, never hand-edited after generation. Notebook cell outputs are stripped before commit by the `nbstripout` pre-commit hook — a notebook only needs to *execute cleanly*, not carry committed outputs.
- **Code conventions (`AGENTS.md`):** ruff + black, line length 100, Python ≥3.11, no `print()` in `src/unfallatlas/` modules (`logging` instead — notebooks may `print`/`display`), all paths via `pathlib.Path`, model artefacts saved under `data/processed/`.
- **Data loading convention:** prefer DuckDB for full-table aggregation; the enriched training frame for A³ is the **existing** cached Parquet built by the U-phase (`data/interim/accidents_with_weather.parquet`, built by `unfallatlas.data.dwd.build_weather_features` and already exercised in `notebooks/02_U_Phase.py` §8.5) — A³ **loads this cache, it does not rebuild it**.

---

## A³ Scope — exact boundary (read this before writing anything)

The previous two phases drifted on scope more than once. This section is the single source of truth for what is and is not A³ work. If a task or a reviewer finds itself doing something not listed under "In scope," stop and flag it — do not silently expand.

### In scope

1. A single `sklearn.pipeline.Pipeline`-compatible preprocessor implementing the U-phase §10 table **exactly as written above** — no new engineered features beyond what §10 already decided (no holiday flags, no weekend flags, no lag features, no H3 cells, no OSM/road-context features — none of these were decided in U, so A³ does not invent them).
2. Baselines (Stufe 0, `PROJEKTPLAN_SETUP.md`): random-guess, majority-class, logistic regression.
3. Tree ensembles (Stufe 1): Random Forest, XGBoost, LightGBM, CatBoost — each trained once with default hyperparameters and once with `class_weight="balanced"` (or the library's equivalent) — 8 configurations total, evaluated on the 2023 validation split.
4. Selecting a single **champion** = the (model, class-weight-setting) combination with the highest validation macro-F1.
5. Imbalance-strategy comparison (Stufe 2), run **only on the champion's base estimator**, on a bounded training subsample (≤ 500,000 rows, stratified by `UKATGEORIE`): SMOTE, ADASYN, threshold moving, and ordinal classification (Frank–Hall decomposition). This yields 4 additional configurations compared against the champion's own validation score — **not** a full cross-product of every strategy against every model family.
6. Hyperparameter tuning with Optuna (bounded: ≤ 40 trials, TPE sampler) of **only** the single best (model, imbalance-strategy) combination that emerges from step 5, on the same bounded subsample, optimising mean validation macro-F1 under `GroupKFold(groups=UJAHR)`.
7. Refitting the tuned winning pipeline on the **full** 2016–2022 training set.
8. Exactly one evaluation on the 2024 test set, reporting macro-F1, recall per class, and the confusion matrix, with an explicit PASS/FAIL statement against the Q-phase thresholds (≥0.55 / ≥0.50).
9. Saving the fitted winning pipeline + a small JSON "model card" (config, metrics, provenance) to `data/processed/`, for the C phase to load without retraining.
10. A model-comparison table (all 8 + 4 + 1 tuned configurations, validation scores) as the central portfolio artefact of this phase.
11. Library code for all of the above, each function covered by a focused pytest test.
12. Updating `AGENTS.md`'s notebook status table, `docs/GLOSSARY.md` with new A³ terms, and the A³ row(s) of `docs/AI TOOL DISCLOSURE.md`.

### Explicitly out of scope (deferred to later phases or dropped)

- **SHAP / any explainability analysis** — this is Phase C's job (`AGENTS.md`: `04_C_Phase.ipynb # Comparison, SHAP, conclusions`). A³ trains an interpretable-by-construction model family; it does not compute or plot SHAP values.
- **Literature-comparison narrative, limitations discussion, final write-up** — Phase C.
- **Streamlit app / any K-phase deployment work.**
- **OSM/Overpass road-context enrichment, H3 hex-bin features, spatial GNNs** — explicitly marked "optional, Stufe 4 / for Bestnote" in `docs/project/PROJEKTPLAN_SETUP.md`; not required for the macro-F1 ≥ 0.55 acceptance target and not part of this plan.
- **Ensembling / stacking multiple models** — explicitly "if time" in the old planning doc; not required to meet acceptance criteria; skipped here. If a future phase wants it, that is a new plan.
- **Full hyperparameter tuning across every model × every imbalance strategy** — only the single winning combination gets tuned (see "In scope" §6), to respect the compute-budget soft constraint.
- **Re-opening the U-phase §10 decision table or the Q-phase metric definitions.** If a task discovers the table is wrong, that is a finding to report to the user, not a silent edit.
- **Editing `04_C_Phase.ipynb`, `app/streamlit_app.py`, or `reports/final_report.md`.**
- **Rebuilding the DWD weather cache or re-running the OSM/Overpass calls** — A³ reads the existing `data/interim/accidents_with_weather.parquet` cache.

---

## File Structure

```
docs/prompts/03_prompts_phase_a3.md      # NEW — the approved-plan prompt artefact (Task 1)
docs/AI TOOL DISCLOSURE.md               # MODIFIED — new Phase A³ row(s) (Task 1)

src/unfallatlas/features/temporal.py     # NEW — cyclic sin/cos encoding (Task 2)
tests/test_temporal.py                   # NEW (Task 2)

src/unfallatlas/features/preprocessing.py # NEW — U§10 ColumnTransformer + data loading/split helpers (Task 3)
tests/test_preprocessing.py              # NEW (Task 3)

src/unfallatlas/models/evaluate.py       # NEW — macro-F1 / recall / acceptance-gate helpers (Task 4)
tests/test_evaluate.py                   # NEW (Task 4)

src/unfallatlas/models/baseline.py       # NEW — random/majority/logreg builders (Task 5)
tests/test_models_baseline.py            # NEW (Task 5)

src/unfallatlas/models/boosting.py       # NEW — RF/XGBoost/LightGBM/CatBoost builders (Task 6)
tests/test_models_boosting.py            # NEW (Task 6)

src/unfallatlas/models/ordinal.py        # NEW — Frank–Hall ordinal classifier (Task 7)
tests/test_models_ordinal.py             # NEW (Task 7)

src/unfallatlas/models/imbalance.py      # NEW — SMOTE/ADASYN/threshold-moving helpers (Task 8)
tests/test_imbalance.py                  # NEW (Task 8)

notebooks/03_A3_Phase.py                 # REWRITTEN (jupytext mirror) — Tasks 9 & 10
notebooks/03_A3_Phase.ipynb              # REGENERATED from the .py mirror — Task 11

AGENTS.md                                # MODIFIED — notebook status table (Task 11)
docs/GLOSSARY.md                         # MODIFIED — new A³ terms (Task 11)

data/processed/a3_best_model.joblib      # NEW (generated by notebook execution, Task 11)
data/processed/a3_model_card.json        # NEW (generated by notebook execution, Task 11)
```

`src/unfallatlas/features/preprocessing.py` is a **new** file not previously listed in `AGENTS.md`'s architecture map (which anticipated `enrich.py` for joins and `spatial.py` for H3/distance features — both out of scope here). It exists because the U§10 ColumnTransformer construction is feature engineering that belongs with `temporal.py`, not with the (unrelated, out-of-scope) join/spatial files. Likewise `models/imbalance.py` is new — imbalance-mitigation helpers are their own responsibility, not a fit for `baseline.py`/`boosting.py`/`ordinal.py`. Task 11 updates `AGENTS.md`'s architecture section to reflect both additions.

---

### Task 1: Write the A³ prompt artefact and update the AI-tool disclosure

**Files:**
- Create: `docs/prompts/03_prompts_phase_a3.md`
- Modify: `docs/AI TOOL DISCLOSURE.md`

**Interfaces:**
- Consumes: this plan document in full (it is the source of truth for the prompt's content).
- Produces: nothing consumed by later tasks — this is a documentation-only task, done first so the disclosure references the approved scope before implementation begins.

- [ ] **Step 1: Write the prompt file**

Create `docs/prompts/03_prompts_phase_a3.md` mirroring the structure of `docs/prompts/02_prompts_phase_u.md` (header → **initial prompt** → primary objective → required repository context → project context from U-phase → **strict A³-phase scope** → important boundary rules → numbered tasks → required outputs → repository organisation guidance → quality requirements → final response format). Use this exact content:

```markdown
# AI Prompt for Phase A³

## Build the A³-Phase from the U-Phase Contract

**Claude Code (Sonnet 5), effort: medium (Anthropic, 2026):**

### Initial prompt

The A³-phase plan was produced with the `superpowers:writing-plans` skill
(Claude Code / superpowers plugin), starting from the following two
verbatim user messages:

> Now please do me a favour now the A Phase writing creation and
> implementation is the next step and i want you to first read all the
> @docs/ and especiall the relevant ones additionally the already existing
> Q and U Phase as it builds upon these 2 then craft a veeery detailed plan
> using /writing-plans where you make AN EXACT SCOPE what the exact things
> are that HAVE to be done in A Phase and what explicitly DONT HAS to go in
> the A PHase as i had issues previously about exactly this in Q and U
> phase with you so make this extra good. Then utilize all the information
> to make a extremly strong A Phase tailored perfectly to my project and
> its context. Then display the plan to me so i can approve it

> IMPORTANT write the finished approved plan as first task inside
> @docs/prompts/ folder as a new prompt file this one for A phase and
> update @"docs/AI TOOL DISCLOSURE.md"

The resulting task-by-task plan lives at
`docs/superpowers/plans/2026-07-01-a3-phase-modelling.md`; this prompt file
is that plan's Task 1 deliverable.

```markdown
You are a senior Machine Learning Engineer, Data Scientist, and repository-quality
documentation architect.

Your task is to implement the **A³-Phase — Algorithm / Adapt / Adjust** of the
QUA³CK process model for this repository.

The **Q-Phase** and **U-Phase already exist** and are the authoritative contracts.
Your job is not to redesign them. Your job is to implement the U-Phase §10
preprocessing decision table, train and compare the models the U-Phase
handed off to A³, tune the winner, and evaluate it once against the
Q-Phase acceptance criteria.

---

# PRIMARY OBJECTIVE

Implement `notebooks/03_A3_Phase.ipynb` (+ paired `.py` mirror) and the
supporting `src/unfallatlas/{features,models}/` library code so that the
project has a working, tested, reproducible modelling pipeline answering:

> "Given the U-Phase's preprocessing contract, which model and imbalance
> strategy best predicts UKATGEORIE, and does the tuned winner clear the
> Q-Phase bar of macro-F1 >= 0.55 and recall(class 1) >= 0.50 on the 2024
> held-out test year?"

---

# REQUIRED REPOSITORY CONTEXT

You MUST read before writing anything:

* `notebooks/01_Q_Phase.ipynb` / `.py` — target, metrics, split, constraints
* `notebooks/02_U_Phase.ipynb` / `.py`, especially §9 (leakage audit) and
  §10 (preprocessing decisions) and §11 (top-4 risks for A³)
* `AGENTS.md` — architecture, conventions, notebook policy
* `docs/GLOSSARY.md` — existing terminology
* `pyproject.toml` — already-declared dependencies (no new ones needed)
* the empty stub files at `src/unfallatlas/features/temporal.py` and
  `src/unfallatlas/models/{baseline,boosting,ordinal,evaluate}.py`

Do not contradict the Q-Phase or U-Phase. If you find a mismatch between
what U-Phase §10 specifies and what the data actually contains, document it
as an A³-Phase finding — do not silently change the U-Phase.

---

# STRICT A³-PHASE SCOPE

Include:

* the U-Phase §10 preprocessing pipeline, implemented exactly as decided
  (cyclic encoding, one-hot, target-mean encoding with smoothing, log/log1p
  + scaling, drops) — no new engineered features beyond that table
* baseline models: random guess, majority class, logistic regression
* tree ensembles: Random Forest, XGBoost, LightGBM, CatBoost, each with a
  default and a class-weighted configuration
* selection of a single champion model by validation macro-F1
* an imbalance-strategy comparison (SMOTE, ADASYN, threshold moving,
  ordinal/Frank-Hall classification) run only on the champion
* bounded Optuna tuning (<= 40 trials) of only the single winning
  (model, strategy) combination
* exactly one evaluation on the 2024 test year
* a saved model artefact + model card for the C-Phase to consume
* a model-comparison table as the central portfolio output
* pytest coverage for every new library function

Do NOT include:

* SHAP or any explainability analysis (Phase C)
* literature-comparison narrative or limitations discussion (Phase C)
* Streamlit app or any deployment work (Phase K)
* OSM/Overpass, H3 hex-bins, or any spatial enrichment beyond what the
  U-Phase cache already contains
* ensembling/stacking of multiple models
* full hyperparameter tuning across every model x every strategy
  combination — only the single winner is tuned
* new dependencies in `pyproject.toml`
* edits to `04_C_Phase.ipynb`, `app/streamlit_app.py`, or
  `reports/final_report.md`
* rebuilding the DWD weather cache

Those belong to later QUA³CK phases or are out of scope entirely.

---

# IMPORTANT BOUNDARY RULES

The A³-Phase may say:

* "The champion model is X; the tuned winner is X + strategy Y."
* "macro-F1 on test-2024 is Z; this [passes/fails] the >= 0.55 threshold."
* "This preprocessing decision follows U-Phase §10 verbatim."

The A³-Phase must NOT say:

* "SHAP shows feature Z is most important." (Phase C)
* "This model is ready to deploy in Streamlit." (Phase K)
* "We should add H3 features here." (out of scope — note it as a future
  idea in the handoff, do not implement it)

When uncertain, defer the topic to C or K instead of expanding A³.

---

# TASKS

Follow `docs/superpowers/plans/2026-07-01-a3-phase-modelling.md` task-by-task.
Each task specifies exact files, function signatures, and test code.

---

# REQUIRED OUTPUTS

1. `src/unfallatlas/features/temporal.py` + `preprocessing.py`, tested
2. `src/unfallatlas/models/{evaluate,baseline,boosting,ordinal,imbalance}.py`, tested
3. `notebooks/03_A3_Phase.ipynb` + paired `.py`, executing end-to-end
4. `data/processed/a3_best_model.joblib` + `a3_model_card.json`
5. Updated `AGENTS.md` notebook status table and architecture section
6. Updated `docs/GLOSSARY.md` with new A³ terms
7. This prompt file and the corresponding `AI TOOL DISCLOSURE.md` row

---

# QUALITY REQUIREMENTS

The A³-Phase must be reproducible, tested, strict about phase boundaries,
and honest about its single test-set evaluation (no re-running the test
set to chase a better number). Every reported metric must come from an
executed cell, not a fabricated value.

---

# FINAL RESPONSE FORMAT

## Summary of Changes

## Key A³-Phase Findings (champion, winning strategy, final test metrics, PASS/FAIL)
```
```

- [ ] **Step 2: Add the A³ row(s) to the disclosure table**

In `docs/AI TOOL DISCLOSURE.md`, insert new rows into the "Detailed overview of the AI tools used in each phase" table, directly after the last `**Phase U**` row (currently ending at what is today line 32) and before the closing `---` (currently line 33). Use `Edit` to replace:

```
| **Phase U** | Claude Code (Sonnet 4.6), effort: medium (Anthropic, 2026) | Final U-Phase polish pass: consistency and visualization cleanup across §1–§9; `GLOSSARY.md` and `README.md` updates; DWD CDC weather enrichment in `src/unfallatlas/data/dwd.py` (station parsing, nearest-station lookup via cKDTree, per-station download, left-join enrichment) plus §8.5–§8.7 weather analysis and §9.4 leakage probe | view [docs/prompts/02_prompts_phase_u.md](docs/prompts/02_prompts_phase_u.md) |

---
```

with:

```
| **Phase U** | Claude Code (Sonnet 4.6), effort: medium (Anthropic, 2026) | Final U-Phase polish pass: consistency and visualization cleanup across §1–§9; `GLOSSARY.md` and `README.md` updates; DWD CDC weather enrichment in `src/unfallatlas/data/dwd.py` (station parsing, nearest-station lookup via cKDTree, per-station download, left-join enrichment) plus §8.5–§8.7 weather analysis and §9.4 leakage probe | view [docs/prompts/02_prompts_phase_u.md](docs/prompts/02_prompts_phase_u.md) |
| **Phase A³** | Claude Code (Sonnet 5), effort: medium (Anthropic, 2026) | A³-Phase scope definition and implementation plan, built with the `superpowers:writing-plans` skill from a two-message user prompt: translating the U-Phase §10 preprocessing contract and top-4 risks into an exact, bounded task plan (baselines, boosting models, imbalance-strategy comparison, Optuna tuning, single test-2024 evaluation) with an explicit in-scope/out-of-scope boundary against Phase C | view [docs/prompts/03_prompts_phase_a3.md](docs/prompts/03_prompts_phase_a3.md) |
| **Phase A³** | Claude Code (Sonnet 5), effort: medium (Anthropic, 2026) | A³-Phase build-out: preprocessing pipeline, baseline/tree/ordinal/imbalance library modules with pytest coverage, `03_A3_Phase.ipynb` (model comparison table, champion selection, Optuna tuning, held-out test evaluation, model-card artefact), A³-to-C handoff | view [docs/prompts/03_prompts_phase_a3.md](docs/prompts/03_prompts_phase_a3.md) |

---
```

- [ ] **Step 3: Commit**

```bash
git add "docs/prompts/03_prompts_phase_a3.md" "docs/AI TOOL DISCLOSURE.md"
git commit -m "docs: add A³-phase prompt artefact and disclosure rows"
```

---

### Task 2: Cyclic temporal encoding

**Files:**
- Create: `src/unfallatlas/features/temporal.py`
- Test: `tests/test_temporal.py`

**Interfaces:**
- Produces: `cyclic_encode(df: pd.DataFrame, column: str, period: int) -> pd.DataFrame` — returns a copy of `df` with `{column}_sin` and `{column}_cos` added.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_temporal.py
import numpy as np
import pandas as pd

from unfallatlas.features.temporal import cyclic_encode


def test_cyclic_encode_adds_sin_cos_columns():
    df = pd.DataFrame({"USTUNDE": [0, 6, 12, 18, 23]})
    out = cyclic_encode(df, "USTUNDE", period=24)
    assert {"USTUNDE_sin", "USTUNDE_cos"}.issubset(out.columns)
    assert len(out) == 5


def test_cyclic_encode_does_not_mutate_input():
    df = pd.DataFrame({"USTUNDE": [0, 12]})
    cyclic_encode(df, "USTUNDE", period=24)
    assert list(df.columns) == ["USTUNDE"]


def test_cyclic_encode_wraps_hour_0_and_24_to_the_same_point():
    df = pd.DataFrame({"USTUNDE": [0, 24]})
    out = cyclic_encode(df, "USTUNDE", period=24)
    np.testing.assert_allclose(out["USTUNDE_sin"].iloc[0], out["USTUNDE_sin"].iloc[1], atol=1e-9)
    np.testing.assert_allclose(out["USTUNDE_cos"].iloc[0], out["USTUNDE_cos"].iloc[1], atol=1e-9)


def test_cyclic_encode_hour_23_is_closer_to_0_than_to_12():
    near = cyclic_encode(pd.DataFrame({"USTUNDE": [23, 0]}), "USTUNDE", period=24)
    far = cyclic_encode(pd.DataFrame({"USTUNDE": [23, 12]}), "USTUNDE", period=24)
    near_dist = np.hypot(
        near["USTUNDE_sin"].iloc[0] - near["USTUNDE_sin"].iloc[1],
        near["USTUNDE_cos"].iloc[0] - near["USTUNDE_cos"].iloc[1],
    )
    far_dist = np.hypot(
        far["USTUNDE_sin"].iloc[0] - far["USTUNDE_sin"].iloc[1],
        far["USTUNDE_cos"].iloc[0] - far["USTUNDE_cos"].iloc[1],
    )
    assert near_dist < far_dist
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_temporal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'unfallatlas.features.temporal'` (or `ImportError: cannot import name 'cyclic_encode'` if the empty file exists but is unpopulated).

- [ ] **Step 3: Write the implementation**

```python
# src/unfallatlas/features/temporal.py
"""Cyclic encoding for periodic time features (U-phase §10 contract).

Only the three columns the U-phase decided on are ever encoded here:
UMONAT (period 12), USTUNDE (period 24), UWOCHENTAG (period 7). This module
does not add holiday flags, weekend flags, or any other temporal feature —
those were not part of the U-phase §10 handover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cyclic_encode(df: pd.DataFrame, column: str, period: int) -> pd.DataFrame:
    """Return a copy of ``df`` with ``{column}_sin`` / ``{column}_cos`` added.

    The circular encoding ensures the model sees hour 23 and hour 0 as
    adjacent rather than 23 apart — the reason U-phase §10 mandates this
    over a raw integer or one-hot encoding for UMONAT/USTUNDE/UWOCHENTAG.
    """
    out = df.copy()
    radians = 2 * np.pi * out[column].astype(float) / period
    out[f"{column}_sin"] = np.sin(radians)
    out[f"{column}_cos"] = np.cos(radians)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_temporal.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/unfallatlas/features/temporal.py tests/test_temporal.py
git commit -m "feat: add cyclic temporal encoding for A3 phase"
```

---

### Task 3: Preprocessing pipeline and data-loading helpers

**Files:**
- Create: `src/unfallatlas/features/preprocessing.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: `cyclic_encode(df, column, period)` from Task 2.
- Produces:
  - `TargetMeanEncoder(BaseEstimator, TransformerMixin)` — `.fit(X, y)`, `.transform(X) -> pd.DataFrame`, columns named `{col}_target_enc`.
  - `CyclicEncoder(BaseEstimator, TransformerMixin)` — `.fit(X, y=None)`, `.transform(X) -> np.ndarray` (2 columns).
  - `build_preprocessor(scale_for_linear: bool = False) -> sklearn.compose.ColumnTransformer`
  - `load_training_frame(base_dir: Path) -> pd.DataFrame`
  - `chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]` (train, val, test)
  - `split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]` (X, y)
  - Module constant `TARGET_COLUMN = "UKATGEORIE"`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preprocessing.py
import numpy as np
import pandas as pd
import pytest

from unfallatlas.features.preprocessing import (
    TargetMeanEncoder,
    build_preprocessor,
    chronological_split,
    split_features_target,
)


def _toy_frame(n=60):
    rng = np.random.default_rng(42)
    years = np.repeat([2016, 2017, 2022, 2023, 2024], n // 5)
    return pd.DataFrame(
        {
            "UJAHR": years,
            "UMONAT": rng.integers(1, 13, n),
            "USTUNDE": rng.integers(0, 24, n),
            "UWOCHENTAG": rng.integers(1, 8, n),
            "UART": rng.integers(0, 10, n),
            "UTYP1": rng.integers(1, 8, n),
            "ULICHTVERH": rng.integers(0, 3, n),
            "STRZUSTAND": rng.integers(0, 3, n),
            "IstRad": rng.integers(0, 2, n),
            "IstPKW": rng.integers(0, 2, n),
            "IstFuss": rng.integers(0, 2, n),
            "IstKrad": rng.integers(0, 2, n),
            "IstGkfz": rng.integers(0, 2, n),
            "IstSonstig": rng.integers(0, 2, n),
            "LON": rng.uniform(6, 15, n),
            "LAT": rng.uniform(47, 55, n),
            "UREGBEZ": rng.integers(1, 5, n),
            "UKREIS": rng.integers(1000, 1050, n),
            "dwd_temp_air_2m": rng.normal(10, 5, n),
            "dwd_precip_mm": rng.exponential(1.0, n),
            "dwd_visibility_m": rng.exponential(5000, n),
            "dwd_wind_speed_ms": rng.normal(3, 1, n),
            "dwd_station_dist_km": rng.uniform(0.1, 40, n),
            "UKATGEORIE": rng.choice([1, 2, 3], n, p=[0.05, 0.25, 0.70]),
        }
    )


def test_chronological_split_respects_year_boundaries():
    df = _toy_frame()
    train, val, test = chronological_split(df)
    assert train["UJAHR"].max() <= 2022
    assert set(val["UJAHR"].unique()) == {2023}
    assert set(test["UJAHR"].unique()) == {2024}
    assert len(train) + len(val) + len(test) == len(df)


def test_split_features_target_drops_year_and_target():
    df = _toy_frame()
    X, y = split_features_target(df)
    assert "UJAHR" not in X.columns
    assert "UKATGEORIE" not in X.columns
    assert y.name == "UKATGEORIE"
    assert set(y.unique()).issubset({1, 2, 3})


def test_target_mean_encoder_smooths_toward_global_mean():
    X = pd.DataFrame({"UKREIS": [1, 1, 1, 2]})
    y = pd.Series([3, 3, 3, 1])  # UKREIS=1 always severity 3, UKREIS=2 only one obs of 1
    enc = TargetMeanEncoder(columns=["UKREIS"], smoothing=10.0)
    enc.fit(X, y)
    out = enc.transform(X)
    # UKREIS=2 has a single observation; smoothing must pull it toward the
    # global mean (2.5) rather than reporting the raw value (1.0).
    assert out["UKREIS_target_enc"].iloc[3] > 1.0
    assert out["UKREIS_target_enc"].iloc[3] < 2.5


def test_target_mean_encoder_unseen_category_gets_global_mean():
    X_train = pd.DataFrame({"UKREIS": [1, 1, 2, 2]})
    y_train = pd.Series([1, 1, 3, 3])
    enc = TargetMeanEncoder(columns=["UKREIS"], smoothing=5.0)
    enc.fit(X_train, y_train)
    out = enc.transform(pd.DataFrame({"UKREIS": [999]}))
    assert out["UKREIS_target_enc"].iloc[0] == pytest.approx(enc.global_mean_)


def test_build_preprocessor_fit_transform_has_no_nans():
    df = _toy_frame()
    train, _, _ = chronological_split(df)
    X, y = split_features_target(train)
    preprocessor = build_preprocessor(scale_for_linear=False)
    transformed = preprocessor.fit_transform(X, y)
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == len(X)


def test_build_preprocessor_scale_for_linear_changes_passthrough_columns():
    df = _toy_frame()
    train, _, _ = chronological_split(df)
    X, y = split_features_target(train)
    tree_pre = build_preprocessor(scale_for_linear=False).fit(X, y)
    linear_pre = build_preprocessor(scale_for_linear=True).fit(X, y)
    tree_out = tree_pre.transform(X)
    linear_out = linear_pre.transform(X)
    # Same number of rows either way; scaling changes values, not row count.
    assert tree_out.shape[0] == linear_out.shape[0] == len(X)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'unfallatlas.features.preprocessing'`

- [ ] **Step 3: Write the implementation**

```python
# src/unfallatlas/features/preprocessing.py
"""U-phase §10 preprocessing contract, implemented as a single sklearn Pipeline.

The U-phase decided; this module implements exactly what
notebooks/02_U_Phase.py §10 specifies — no additional engineered features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from unfallatlas.features.temporal import cyclic_encode

TARGET_COLUMN = "UKATGEORIE"
SPLIT_YEAR_COLUMN = "UJAHR"
NON_FEATURE_COLUMNS = ["OBJECTID", "UGEMEINDE"]

CYCLIC_COLUMNS = {"UMONAT": 12, "USTUNDE": 24, "UWOCHENTAG": 7}
ONEHOT_COLUMNS = ["UART", "UTYP1", "ULICHTVERH", "STRZUSTAND"]
TARGET_ENCODED_COLUMNS = ["UREGBEZ", "UKREIS"]
PASSTHROUGH_COLUMNS = [
    "IstRad", "IstPKW", "IstFuss", "IstKrad", "IstGkfz", "IstSonstig", "LON", "LAT",
]
LOG1P_COLUMNS = ["dwd_precip_mm", "dwd_visibility_m"]
LOG_COLUMNS = ["dwd_station_dist_km"]
PLAIN_NUMERIC_COLUMNS = ["dwd_temp_air_2m", "dwd_wind_speed_ms"]


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """Mean-target encoding with additive smoothing (U-phase §10).

    Encodes each category as the smoothed mean of the *numeric* target code
    (1/2/3) observed for that category in the training fold — consistent
    with the Q-phase §5 note that the three classes have a natural order.
    Fit on training data only; unseen categories at transform time receive
    the global training mean.
    """

    def __init__(self, columns: list[str], smoothing: float = 10.0):
        self.columns = columns
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        y_numeric = pd.Series(np.asarray(y), index=X.index, dtype=float)
        self.global_mean_ = float(y_numeric.mean())
        self.mappings_: dict[str, pd.Series] = {}
        for col in self.columns:
            grp = y_numeric.groupby(X[col], observed=True)
            counts = grp.count()
            means = grp.mean()
            self.mappings_[col] = (
                counts * means + self.smoothing * self.global_mean_
            ) / (counts + self.smoothing)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in self.columns:
            out[f"{col}_target_enc"] = (
                X[col].map(self.mappings_[col]).fillna(self.global_mean_).astype(float)
            )
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{col}_target_enc" for col in self.columns])


class CyclicEncoder(BaseEstimator, TransformerMixin):
    """sklearn-compatible wrapper around ``cyclic_encode`` for a ColumnTransformer."""

    def __init__(self, period: int):
        self.period = period

    def fit(self, X: pd.DataFrame, y=None):
        self.column_ = X.columns[0]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        encoded = cyclic_encode(X, self.column_, self.period)
        return encoded[[f"{self.column_}_sin", f"{self.column_}_cos"]].to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.column_}_sin", f"{self.column_}_cos"])


def build_preprocessor(scale_for_linear: bool = False) -> ColumnTransformer:
    """Build the ColumnTransformer implementing U-phase §10 verbatim.

    scale_for_linear=True additionally scales LON/LAT and the binary
    transport-mode flags — required by the Logistic Regression baseline;
    U-phase §10 marks this scaling "only for distance-based baselines,"
    which tree models do not need.
    """
    transformers = [
        (f"cyclic_{col}", CyclicEncoder(period=period), [col])
        for col, period in CYCLIC_COLUMNS.items()
    ]

    transformers.append(
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ONEHOT_COLUMNS)
    )
    transformers.append(
        ("target_enc", TargetMeanEncoder(columns=TARGET_ENCODED_COLUMNS), TARGET_ENCODED_COLUMNS)
    )

    log1p_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    transformers.append(("log1p_cols", log1p_pipeline, LOG1P_COLUMNS))

    log_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(lambda a: np.log(a + 1e-6), feature_names_out="one-to-one")),
        ]
    )
    transformers.append(("log_cols", log_pipeline, LOG_COLUMNS))

    plain_numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    transformers.append(("plain_numeric", plain_numeric_pipeline, PLAIN_NUMERIC_COLUMNS))

    if scale_for_linear:
        transformers.append(("passthrough_scaled", StandardScaler(), PASSTHROUGH_COLUMNS))
    else:
        transformers.append(("passthrough", "passthrough", PASSTHROUGH_COLUMNS))

    return ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=False
    )


def load_training_frame(base_dir: Path) -> pd.DataFrame:
    """Load the DWD-enriched accidents frame built by the U-phase.

    Reuses the cache from ``unfallatlas.data.dwd.build_weather_features``
    (``data/interim/accidents_with_weather.parquet``). A³ does not rebuild
    this cache — raises if it is missing.
    """
    cache = base_dir / "data" / "interim" / "accidents_with_weather.parquet"
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} not found. Run notebooks/02_U_Phase.ipynb §8.5 first to "
            "build the weather-enriched cache."
        )
    df = pd.read_parquet(cache)
    return df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])


def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train 2016-2022 / val 2023 / test 2024 — Q-phase §6, U-phase §9.2."""
    train = df[df[SPLIT_YEAR_COLUMN] <= 2022].reset_index(drop=True)
    val = df[df[SPLIT_YEAR_COLUMN] == 2023].reset_index(drop=True)
    test = df[df[SPLIT_YEAR_COLUMN] == 2024].reset_index(drop=True)
    return train, val, test


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop UJAHR/target from the feature frame; return (X, y)."""
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN, SPLIT_YEAR_COLUMN])
    return X, y
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/unfallatlas/features/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: implement U-phase §10 preprocessing pipeline"
```

---

### Task 4: Evaluation metrics and the acceptance gate

**Files:**
- Create: `src/unfallatlas/models/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces:
  - `MACRO_F1_THRESHOLD = 0.55`, `RECALL_CLASS_1_THRESHOLD = 0.50` (module constants)
  - `macro_f1(y_true, y_pred) -> float`
  - `recall_for_class(y_true, y_pred, target_class: int) -> float`
  - `evaluate_predictions(y_true, y_pred) -> dict` (keys: `macro_f1`, `recall_class_1`, `recall_class_2`, `recall_class_3`, `confusion_matrix`)
  - `meets_acceptance_criteria(metrics: dict) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import numpy as np

from unfallatlas.models.evaluate import (
    MACRO_F1_THRESHOLD,
    RECALL_CLASS_1_THRESHOLD,
    evaluate_predictions,
    macro_f1,
    meets_acceptance_criteria,
    recall_for_class,
)


def test_macro_f1_perfect_predictions_is_one():
    y = np.array([1, 2, 3, 1, 2, 3])
    assert macro_f1(y, y) == 1.0


def test_recall_for_class_1_zero_when_never_predicted():
    y_true = np.array([1, 1, 2, 3, 3])
    y_pred = np.array([3, 3, 2, 3, 3])  # class 1 never predicted
    assert recall_for_class(y_true, y_pred, target_class=1) == 0.0


def test_evaluate_predictions_returns_all_expected_keys():
    y = np.array([1, 2, 3, 1, 2, 3])
    metrics = evaluate_predictions(y, y)
    assert set(metrics) == {
        "macro_f1", "recall_class_1", "recall_class_2", "recall_class_3", "confusion_matrix",
    }
    assert metrics["macro_f1"] == 1.0
    assert metrics["recall_class_1"] == 1.0


def test_meets_acceptance_criteria_requires_both_thresholds():
    passing = {"macro_f1": MACRO_F1_THRESHOLD, "recall_class_1": RECALL_CLASS_1_THRESHOLD}
    failing_macro = {"macro_f1": MACRO_F1_THRESHOLD - 0.01, "recall_class_1": RECALL_CLASS_1_THRESHOLD}
    failing_recall = {"macro_f1": MACRO_F1_THRESHOLD, "recall_class_1": RECALL_CLASS_1_THRESHOLD - 0.01}
    assert meets_acceptance_criteria(passing) is True
    assert meets_acceptance_criteria(failing_macro) is False
    assert meets_acceptance_criteria(failing_recall) is False


def test_meets_acceptance_criteria_majority_class_baseline_fails():
    # Majority-class prediction: everything predicted as class 3.
    y_true = np.array([1] * 10 + [2] * 180 + [3] * 810)
    y_pred = np.array([3] * len(y_true))
    metrics = evaluate_predictions(y_true, y_pred)
    assert meets_acceptance_criteria(metrics) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'unfallatlas.models.evaluate'`

- [ ] **Step 3: Write the implementation**

```python
# src/unfallatlas/models/evaluate.py
"""Evaluation metrics and the Q-phase §8 acceptance gate."""

from __future__ import annotations

from sklearn.metrics import confusion_matrix, f1_score, recall_score

MACRO_F1_THRESHOLD = 0.55
RECALL_CLASS_1_THRESHOLD = 0.50


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def recall_for_class(y_true, y_pred, target_class: int) -> float:
    return float(recall_score(y_true, y_pred, labels=[target_class], average="macro"))


def evaluate_predictions(y_true, y_pred) -> dict:
    """Metrics reported for every model/strategy row in the A³ comparison table."""
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "recall_class_1": recall_for_class(y_true, y_pred, target_class=1),
        "recall_class_2": recall_for_class(y_true, y_pred, target_class=2),
        "recall_class_3": recall_for_class(y_true, y_pred, target_class=3),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[1, 2, 3]).tolist(),
    }


def meets_acceptance_criteria(metrics: dict) -> bool:
    """Q-phase §8 acceptance gate: macro-F1 >= 0.55 AND recall(class 1) >= 0.50."""
    return (
        metrics["macro_f1"] >= MACRO_F1_THRESHOLD
        and metrics["recall_class_1"] >= RECALL_CLASS_1_THRESHOLD
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/unfallatlas/models/evaluate.py tests/test_evaluate.py
git commit -m "feat: add A3 evaluation metrics and acceptance gate"
```

---

### Task 5: Baseline model builders

**Files:**
- Create: `src/unfallatlas/models/baseline.py`
- Test: `tests/test_models_baseline.py`

**Interfaces:**
- Consumes: `build_preprocessor(scale_for_linear=...)` from Task 3.
- Produces:
  - `build_random_guess_classifier() -> sklearn.dummy.DummyClassifier`
  - `build_majority_class_classifier() -> sklearn.dummy.DummyClassifier`
  - `build_logreg_pipeline(preprocessor) -> sklearn.pipeline.Pipeline`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_baseline.py
import numpy as np
import pandas as pd

from unfallatlas.features.preprocessing import build_preprocessor
from unfallatlas.models.baseline import (
    build_logreg_pipeline,
    build_majority_class_classifier,
    build_random_guess_classifier,
)


def _toy_X_y(n=80):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "UMONAT": rng.integers(1, 13, n),
            "USTUNDE": rng.integers(0, 24, n),
            "UWOCHENTAG": rng.integers(1, 8, n),
            "UART": rng.integers(0, 10, n),
            "UTYP1": rng.integers(1, 8, n),
            "ULICHTVERH": rng.integers(0, 3, n),
            "STRZUSTAND": rng.integers(0, 3, n),
            "IstRad": rng.integers(0, 2, n),
            "IstPKW": rng.integers(0, 2, n),
            "IstFuss": rng.integers(0, 2, n),
            "IstKrad": rng.integers(0, 2, n),
            "IstGkfz": rng.integers(0, 2, n),
            "IstSonstig": rng.integers(0, 2, n),
            "LON": rng.uniform(6, 15, n),
            "LAT": rng.uniform(47, 55, n),
            "UREGBEZ": rng.integers(1, 5, n),
            "UKREIS": rng.integers(1000, 1050, n),
            "dwd_temp_air_2m": rng.normal(10, 5, n),
            "dwd_precip_mm": rng.exponential(1.0, n),
            "dwd_visibility_m": rng.exponential(5000, n),
            "dwd_wind_speed_ms": rng.normal(3, 1, n),
            "dwd_station_dist_km": rng.uniform(0.1, 40, n),
        }
    )
    y = pd.Series(rng.choice([1, 2, 3], n, p=[0.1, 0.3, 0.6]))
    return X, y


def test_majority_class_classifier_always_predicts_the_mode():
    X, y = _toy_X_y()
    clf = build_majority_class_classifier()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(preds) == {y.mode().iloc[0]}


def test_random_guess_classifier_predicts_within_known_classes():
    X, y = _toy_X_y()
    clf = build_random_guess_classifier()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(preds).issubset({1, 2, 3})


def test_logreg_pipeline_fits_and_predicts_known_labels():
    X, y = _toy_X_y()
    preprocessor = build_preprocessor(scale_for_linear=True)
    pipe = build_logreg_pipeline(preprocessor)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({1, 2, 3})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'unfallatlas.models.baseline'`

- [ ] **Step 3: Write the implementation**

```python
# src/unfallatlas/models/baseline.py
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
                    multi_class="multinomial",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models_baseline.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/unfallatlas/models/baseline.py tests/test_models_baseline.py
git commit -m "feat: add A3 baseline model builders"
```

---

### Task 6: Boosting/tree ensemble builders

**Files:**
- Create: `src/unfallatlas/models/boosting.py`
- Test: `tests/test_models_boosting.py`

**Interfaces:**
- Consumes: `build_preprocessor(scale_for_linear=...)` from Task 3.
- Produces:
  - `build_random_forest_pipeline(preprocessor, class_weight="balanced") -> Pipeline`
  - `build_xgboost_pipeline(preprocessor) -> Pipeline`
  - `build_lightgbm_pipeline(preprocessor, class_weight="balanced") -> Pipeline`
  - `build_catboost_pipeline(preprocessor, class_weights=None) -> Pipeline`

**Known gotcha to test for explicitly:** the target labels are `{1, 2, 3}`, not `{0, 1, 2}`. Some gradient-boosting libraries have historically required zero-indexed class labels. Every test below asserts `set(preds).issubset({1, 2, 3})` specifically to catch this — if a test fails on that assertion, the fix is to inspect whether the library's sklearn wrapper remaps labels internally (it does for the versions pinned in `pyproject.toml`, but do not assume — let the test prove it).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_boosting.py
import numpy as np
import pandas as pd

from unfallatlas.features.preprocessing import build_preprocessor
from unfallatlas.models.boosting import (
    build_catboost_pipeline,
    build_lightgbm_pipeline,
    build_random_forest_pipeline,
    build_xgboost_pipeline,
)


def _toy_X_y(n=120):
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {
            "UMONAT": rng.integers(1, 13, n),
            "USTUNDE": rng.integers(0, 24, n),
            "UWOCHENTAG": rng.integers(1, 8, n),
            "UART": rng.integers(0, 10, n),
            "UTYP1": rng.integers(1, 8, n),
            "ULICHTVERH": rng.integers(0, 3, n),
            "STRZUSTAND": rng.integers(0, 3, n),
            "IstRad": rng.integers(0, 2, n),
            "IstPKW": rng.integers(0, 2, n),
            "IstFuss": rng.integers(0, 2, n),
            "IstKrad": rng.integers(0, 2, n),
            "IstGkfz": rng.integers(0, 2, n),
            "IstSonstig": rng.integers(0, 2, n),
            "LON": rng.uniform(6, 15, n),
            "LAT": rng.uniform(47, 55, n),
            "UREGBEZ": rng.integers(1, 5, n),
            "UKREIS": rng.integers(1000, 1050, n),
            "dwd_temp_air_2m": rng.normal(10, 5, n),
            "dwd_precip_mm": rng.exponential(1.0, n),
            "dwd_visibility_m": rng.exponential(5000, n),
            "dwd_wind_speed_ms": rng.normal(3, 1, n),
            "dwd_station_dist_km": rng.uniform(0.1, 40, n),
        }
    )
    y = pd.Series(rng.choice([1, 2, 3], n, p=[0.1, 0.3, 0.6]))
    return X, y


def test_random_forest_pipeline_predicts_known_labels():
    X, y = _toy_X_y()
    pipe = build_random_forest_pipeline(build_preprocessor(), class_weight="balanced")
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(preds).issubset({1, 2, 3})


def test_xgboost_pipeline_predicts_known_labels_not_zero_indexed():
    X, y = _toy_X_y()
    pipe = build_xgboost_pipeline(build_preprocessor())
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(preds).issubset({1, 2, 3})


def test_lightgbm_pipeline_predicts_known_labels():
    X, y = _toy_X_y()
    pipe = build_lightgbm_pipeline(build_preprocessor(), class_weight="balanced")
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(preds).issubset({1, 2, 3})


def test_catboost_pipeline_predicts_known_labels():
    X, y = _toy_X_y()
    pipe = build_catboost_pipeline(build_preprocessor())
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(np.asarray(preds).ravel()).issubset({1, 2, 3})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models_boosting.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'unfallatlas.models.boosting'`

- [ ] **Step 3: Write the implementation**

```python
# src/unfallatlas/models/boosting.py
"""Stufe 1 tree ensembles — docs/project/PROJEKTPLAN_SETUP.md ML-Roadmap."""

from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def build_random_forest_pipeline(preprocessor, class_weight: str | dict | None = "balanced") -> Pipeline:
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
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    objective="multi:softprob",
                    num_class=3,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="mlogloss",
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models_boosting.py -v`
Expected: PASS (4 passed). If `test_xgboost_pipeline_predicts_known_labels_not_zero_indexed` fails with predictions in `{0, 1, 2}`, add a label round-trip: wrap `XGBClassifier` fit/predict with a `-1` shift before fit and `+1` after predict, and add a regression test locking in the shift before moving on.

- [ ] **Step 5: Commit**

```bash
git add src/unfallatlas/models/boosting.py tests/test_models_boosting.py
git commit -m "feat: add A3 boosting model builders"
```

---

### Task 7: Ordinal classification (Frank–Hall decomposition)

**Files:**
- Create: `src/unfallatlas/models/ordinal.py`
- Test: `tests/test_models_ordinal.py`

**Interfaces:**
- Produces:
  - `OrdinalClassifier(BaseEstimator, ClassifierMixin)` — `.fit(X, y)`, `.predict_proba(X) -> np.ndarray`, `.predict(X) -> np.ndarray`, attribute `.classes_`
  - `build_ordinal_pipeline(preprocessor, base_estimator) -> Pipeline`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_ordinal.py
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from unfallatlas.models.ordinal import OrdinalClassifier


def _ordered_toy_data(n_per_class=40, seed=0):
    """1-D feature strongly separating the three ordered classes."""
    rng = np.random.default_rng(seed)
    X1 = rng.normal(0, 1, (n_per_class, 1))
    X2 = rng.normal(5, 1, (n_per_class, 1))
    X3 = rng.normal(10, 1, (n_per_class, 1))
    X = np.vstack([X1, X2, X3])
    y = np.array([1] * n_per_class + [2] * n_per_class + [3] * n_per_class)
    return X, y


def test_ordinal_classifier_requires_at_least_three_classes():
    clf = OrdinalClassifier(base_estimator=LogisticRegression())
    with pytest.raises(ValueError):
        clf.fit(np.array([[0], [1]]), np.array([1, 2]))


def test_ordinal_classifier_fits_k_minus_1_binary_estimators():
    X, y = _ordered_toy_data()
    clf = OrdinalClassifier(base_estimator=LogisticRegression())
    clf.fit(X, y)
    assert len(clf.binary_estimators_) == 2  # 3 classes -> 2 thresholds
    assert list(clf.classes_) == [1, 2, 3]


def test_ordinal_classifier_predict_proba_rows_sum_to_one():
    X, y = _ordered_toy_data()
    clf = OrdinalClassifier(base_estimator=LogisticRegression())
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_ordinal_classifier_recovers_well_separated_classes():
    X, y = _ordered_toy_data()
    clf = OrdinalClassifier(base_estimator=LogisticRegression())
    clf.fit(X, y)
    preds = clf.predict(X)
    accuracy = (preds == y).mean()
    assert accuracy > 0.9  # classes are trivially separable by construction
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models_ordinal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'unfallatlas.models.ordinal'`

- [ ] **Step 3: Write the implementation**

```python
# src/unfallatlas/models/ordinal.py
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
        cum_probs = np.column_stack(
            [np.zeros(n)]
            + [est.predict_proba(X)[:, 1] for est in self.binary_estimators_]
            + [np.ones(n)]
        )
        # cum_probs[:, i] = P(y > classes_[i-1]); differencing recovers P(y == classes_[i-1]).
        class_probs = np.clip(np.diff(cum_probs, axis=1), 0, None)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models_ordinal.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/unfallatlas/models/ordinal.py tests/test_models_ordinal.py
git commit -m "feat: add Frank-Hall ordinal classifier for A3"
```

---

### Task 8: Imbalance-strategy helpers (SMOTE, ADASYN, threshold moving)

**Files:**
- Create: `src/unfallatlas/models/imbalance.py`
- Test: `tests/test_imbalance.py`

**Interfaces:**
- Produces:
  - `resample_smote(X, y, random_state=42) -> tuple[X_resampled, y_resampled]`
  - `resample_adasyn(X, y, random_state=42) -> tuple[X_resampled, y_resampled]`
  - `balanced_sample_weight(y) -> np.ndarray` (per-row weights for XGBoost's `sample_weight=`)
  - `find_best_threshold_for_class(y_true, y_proba: np.ndarray, classes: list[int], target_class: int) -> float`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_imbalance.py
import numpy as np
import pandas as pd
import pytest

from unfallatlas.models.imbalance import (
    balanced_sample_weight,
    find_best_threshold_for_class,
    resample_adasyn,
    resample_smote,
)


def _imbalanced_toy_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice([1, 2, 3], n, p=[0.05, 0.25, 0.70])
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    return X, y


def test_resample_smote_balances_class_counts():
    X, y = _imbalanced_toy_data()
    X_res, y_res = resample_smote(X, y)
    counts = pd.Series(y_res).value_counts()
    # SMOTE (default 'auto' strategy) balances every class to the majority count.
    assert counts.nunique() == 1
    assert len(y_res) >= len(y)


def test_resample_adasyn_increases_minority_share():
    X, y = _imbalanced_toy_data()
    original_minority_share = (y == 1).mean()
    X_res, y_res = resample_adasyn(X, y)
    resampled_minority_share = (np.asarray(y_res) == 1).mean()
    assert resampled_minority_share > original_minority_share


def test_balanced_sample_weight_upweights_rare_classes():
    y = np.array([1, 2, 2, 3, 3, 3])
    weights = balanced_sample_weight(y)
    assert len(weights) == len(y)
    # Class 1 (rarest) must receive a strictly higher weight than class 3 (most common).
    assert weights[0] > weights[-1]


def test_find_best_threshold_for_class_improves_on_default_half():
    # 10 rows; class 1 is rare (2 rows) with high probability but a naive
    # argmax at threshold 0.5 for the *other* classes would bury it.
    y_true = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 3])
    y_proba = np.array(
        [
            [0.4, 0.35, 0.25],
            [0.45, 0.3, 0.25],
            [0.1, 0.7, 0.2],
            [0.1, 0.7, 0.2],
            [0.1, 0.7, 0.2],
            [0.05, 0.15, 0.8],
            [0.05, 0.15, 0.8],
            [0.05, 0.15, 0.8],
            [0.05, 0.15, 0.8],
            [0.05, 0.15, 0.8],
        ]
    )
    threshold = find_best_threshold_for_class(y_true, y_proba, classes=[1, 2, 3], target_class=1)
    assert 0.0 < threshold < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_imbalance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'unfallatlas.models.imbalance'`

- [ ] **Step 3: Write the implementation**

```python
# src/unfallatlas/models/imbalance.py
"""Class-imbalance mitigation strategies compared on the A³ champion model.

U-phase §10 menu: class_weight='balanced' is handled directly inside the
model constructors in baseline.py/boosting.py. This module implements the
three remaining strategies: SMOTE, ADASYN, and threshold moving.
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight


def resample_smote(X, y, random_state: int = 42):
    """SMOTE (U-phase §10 menu item) — synthesises minority-class samples."""
    return SMOTE(random_state=random_state).fit_resample(X, y)


def resample_adasyn(X, y, random_state: int = 42):
    """ADASYN (U-phase §10 menu item) — adaptive synthetic oversampling."""
    return ADASYN(random_state=random_state).fit_resample(X, y)


def balanced_sample_weight(y) -> np.ndarray:
    """Per-row weights making every class contribute equally to the loss.

    Used for XGBoost's ``sample_weight=`` argument at fit time, since
    XGBClassifier has no ``class_weight`` parameter.
    """
    return compute_sample_weight(class_weight="balanced", y=y)


def find_best_threshold_for_class(
    y_true, y_proba: np.ndarray, classes: list[int], target_class: int
) -> float:
    """Threshold-moving (U-phase §10 menu item): sweep the decision threshold
    for ``target_class`` and return the value maximising macro-F1.

    ``y_proba`` columns must align with ``classes`` order (as from a
    fitted estimator's ``.classes_``). Predictions falling below the swept
    threshold fall back to whichever of the *other* classes has the
    highest probability among themselves.
    """
    classes = list(classes)
    target_idx = classes.index(target_class)
    other_classes = np.array([c for i, c in enumerate(classes) if i != target_idx])
    other_proba = np.delete(y_proba, target_idx, axis=1)

    best_threshold, best_score = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        fallback = other_classes[np.argmax(other_proba, axis=1)]
        y_pred = np.where(y_proba[:, target_idx] >= threshold, target_class, fallback)
        score = f1_score(y_true, y_pred, average="macro")
        if score > best_score:
            best_score, best_threshold = score, threshold
    return float(best_threshold)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_imbalance.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/unfallatlas/models/imbalance.py tests/test_imbalance.py
git commit -m "feat: add SMOTE/ADASYN/threshold-moving helpers for A3"
```

---

### Task 9: Notebook Part 1 — setup, preprocessing demo, baselines, tree models, champion selection

**Files:**
- Modify (rewrite): `notebooks/03_A3_Phase.py` (jupytext percent-format mirror; the paired `.ipynb` is regenerated in Task 11)

**Interfaces:**
- Consumes: everything produced by Tasks 2–6 (`cyclic_encode`, `build_preprocessor`, `load_training_frame`, `chronological_split`, `split_features_target`, `evaluate_predictions`, `meets_acceptance_criteria`, all `build_*_pipeline` functions from `baseline.py` and `boosting.py`).
- Produces: an in-notebook `comparison_rows: list[dict]` and a chosen `champion_name: str` / `champion_pipeline` that Task 10 continues from.

- [ ] **Step 1: Replace the notebook stub content**

Replace the entire contents of `notebooks/03_A3_Phase.py` (currently the 19-line `# TODO` stub) with the following. This mirrors the header/provenance/position-table conventions of `notebooks/02_U_Phase.py` §0.

```python
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
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import GroupKFold

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
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# %%
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
)
from unfallatlas.models.evaluate import evaluate_predictions, meets_acceptance_criteria
from unfallatlas.models.imbalance import balanced_sample_weight

# %% [markdown]
# ## 1 — Load the U-phase cache and apply the chronological split
#
# A³ does not rebuild the DWD-enriched cache — it reads exactly what
# `notebooks/02_U_Phase.ipynb` §8.5 already produced.

# %%
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
    print(f"{name:35s} macro-F1={metrics['macro_f1']:.3f}  recall(1)={metrics['recall_class_1']:.3f}")


random_guess = build_random_guess_classifier().fit(X_train, y_train)
_score_on_validation("random_guess", random_guess)

majority_class = build_majority_class_classifier().fit(X_train, y_train)
_score_on_validation("majority_class", majority_class)

logreg = build_logreg_pipeline(linear_preprocessor).fit(X_train, y_train)
_score_on_validation("logistic_regression", logreg)

# %% [markdown]
# ## 4 — Stufe 1: tree ensembles, default and class-weighted
#
# Each of Random Forest, XGBoost, LightGBM, and CatBoost is trained twice:
# once with library defaults, once with class-weighting applied — 8
# configurations, all scored once against the 2023 validation split.

# %%
rf_default = build_random_forest_pipeline(tree_preprocessor, class_weight=None).fit(X_train, y_train)
_score_on_validation("random_forest_default", rf_default)

rf_balanced = build_random_forest_pipeline(tree_preprocessor, class_weight="balanced").fit(X_train, y_train)
_score_on_validation("random_forest_balanced", rf_balanced)

xgb_default = build_xgboost_pipeline(tree_preprocessor).fit(X_train, y_train)
_score_on_validation("xgboost_default", xgb_default)

xgb_weights = balanced_sample_weight(y_train)
xgb_balanced = build_xgboost_pipeline(tree_preprocessor)
xgb_balanced.fit(X_train, y_train, classify__sample_weight=xgb_weights)
_score_on_validation("xgboost_balanced", xgb_balanced)

lgbm_default = build_lightgbm_pipeline(tree_preprocessor, class_weight=None).fit(X_train, y_train)
_score_on_validation("lightgbm_default", lgbm_default)

lgbm_balanced = build_lightgbm_pipeline(tree_preprocessor, class_weight="balanced").fit(X_train, y_train)
_score_on_validation("lightgbm_balanced", lgbm_balanced)

catboost_default = build_catboost_pipeline(tree_preprocessor).fit(X_train, y_train)
_score_on_validation("catboost_default", catboost_default)

train_class_counts = y_train.value_counts()
catboost_weights = [len(y_train) / (3 * train_class_counts[c]) for c in [1, 2, 3]]
catboost_balanced = build_catboost_pipeline(tree_preprocessor, class_weights=catboost_weights)
catboost_balanced.fit(X_train, y_train)
_score_on_validation("catboost_balanced", catboost_balanced)

fitted_models = {
    "random_forest_default": rf_default,
    "random_forest_balanced": rf_balanced,
    "xgboost_default": xgb_default,
    "xgboost_balanced": xgb_balanced,
    "lightgbm_default": lgbm_default,
    "lightgbm_balanced": lgbm_balanced,
    "catboost_default": catboost_default,
    "catboost_balanced": catboost_balanced,
}

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
```

- [ ] **Step 2: Materialize and execute the notebook for this part**

Run:
```bash
uv run jupytext --to notebook notebooks/03_A3_Phase.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_A3_Phase.ipynb
```
Expected: exits 0, no exception cells. If `data/interim/accidents_with_weather.parquet` is missing, first run `notebooks/02_U_Phase.ipynb` §8.5 to build it (per `load_training_frame`'s error message).

- [ ] **Step 3: Commit**

```bash
uv run jupytext --sync notebooks/03_A3_Phase.ipynb
git add notebooks/03_A3_Phase.ipynb notebooks/03_A3_Phase.py
git commit -m "feat: A3 notebook part 1 - baselines, tree models, champion selection"
```

---

### Task 10: Notebook Part 2 — imbalance comparison, Optuna tuning, final test evaluation, handoff

**Files:**
- Modify: `notebooks/03_A3_Phase.py` (append to the file created in Task 9)

**Interfaces:**
- Consumes: `champion_name`, `champion_pipeline`, `comparison_rows`, `X_train`/`y_train`/`X_val`/`y_val`/`X_test`/`y_test`, `tree_preprocessor`, `cv_groups` from Task 9; `resample_smote`, `resample_adasyn`, `find_best_threshold_for_class` from Task 8; `build_ordinal_pipeline`, `OrdinalClassifier` from Task 7.
- Produces: `data/processed/a3_best_model.joblib`, `data/processed/a3_model_card.json`.

- [ ] **Step 1: Append the imbalance-strategy comparison section**

Append to `notebooks/03_A3_Phase.py`:

```python
# %% [markdown]
# ## 6 — Imbalance-strategy comparison (champion only)
#
# U-phase §10 menu, compared only on the champion's base estimator, on a
# stratified subsample capped at 500,000 training rows (compute-budget soft
# constraint, Q-phase §9). Class weights are already reflected by whichever
# of `{champion_name}` won in §5 — this section adds SMOTE, ADASYN,
# threshold moving, and ordinal classification on top of the *unweighted*
# variant of the same model family, so all five configurations are
# comparable on equal footing.

# %%
from unfallatlas.models.imbalance import find_best_threshold_for_class, resample_adasyn, resample_smote
from unfallatlas.models.ordinal import build_ordinal_pipeline

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
print(f"Imbalance-comparison subsample: {len(X_train_sub):,} rows (cap={SUBSAMPLE_CAP:,}).")

# %%
champion_builder = {
    "random_forest": lambda pre, **kw: build_random_forest_pipeline(pre, **kw),
    "xgboost": lambda pre, **kw: build_xgboost_pipeline(pre),
    "lightgbm": lambda pre, **kw: build_lightgbm_pipeline(pre, **kw),
    "catboost": lambda pre, **kw: build_catboost_pipeline(pre),
}
champion_family = champion_name.split("_default")[0].split("_balanced")[0]
build_champion = champion_builder[champion_family]

# %%
# 6a: SMOTE
X_smote, y_smote = resample_smote(X_train_sub, y_train_sub)
model_smote = build_champion(tree_preprocessor).fit(X_smote, y_smote)
_score_on_validation(f"{champion_family}_smote", model_smote)

# %%
# 6b: ADASYN
X_adasyn, y_adasyn = resample_adasyn(X_train_sub, y_train_sub)
model_adasyn = build_champion(tree_preprocessor).fit(X_adasyn, y_adasyn)
_score_on_validation(f"{champion_family}_adasyn", model_adasyn)

# %%
# 6c: threshold moving (post-hoc on the unweighted champion-family model)
model_unweighted = build_champion(tree_preprocessor).fit(X_train_sub, y_train_sub)
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
comparison_rows.append({"model": f"{champion_family}_threshold_moving", **threshold_metrics})
print(f"Best threshold for class 1: {best_threshold:.2f}")

# %%
# 6d: ordinal classification (Frank-Hall) using the champion family's own estimator as base
from sklearn.base import clone

base_estimator_for_ordinal = clone(build_champion(tree_preprocessor).named_steps["classify"])
ordinal_pipeline = build_ordinal_pipeline(tree_preprocessor, base_estimator_for_ordinal)
ordinal_pipeline.fit(X_train_sub, y_train_sub)
_score_on_validation(f"{champion_family}_ordinal", ordinal_pipeline)

# %%
comparison_df = pd.DataFrame(comparison_rows)
strategy_rows = comparison_df[comparison_df["model"].str.startswith(champion_family)]
winning_strategy_row = strategy_rows.sort_values("macro_f1", ascending=False).iloc[0]
print(f"Winning (model, strategy) combination: {winning_strategy_row['model']}")
strategy_rows.sort_values("macro_f1", ascending=False)

# %% [markdown]
# ## 7 — Hyperparameter tuning (Optuna, winning combination only)
#
# Bounded to 40 trials on the same subsample, optimising mean validation
# macro-F1 across the year-grouped folds from §2. Only the single winning
# (model, strategy) combination from §6 is tuned — not every configuration
# trained so far (Q-phase §9 compute-budget constraint).

# %%
import optuna
from sklearn.model_selection import cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

PARAM_SPACES = {
    "random_forest": lambda trial: {
        "classify__n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "classify__max_depth": trial.suggest_int("max_depth", 4, 20),
        "classify__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    },
    "xgboost": lambda trial: {
        "classify__n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "classify__max_depth": trial.suggest_int("max_depth", 3, 10),
        "classify__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    },
    "lightgbm": lambda trial: {
        "classify__n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "classify__num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "classify__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    },
    "catboost": lambda trial: {
        "classify__iterations": trial.suggest_int("iterations", 100, 500),
        "classify__depth": trial.suggest_int("depth", 3, 10),
        "classify__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    },
}


n_groups_sub = min(5, years_sub.nunique())


def objective(trial: optuna.Trial) -> float:
    params = PARAM_SPACES[champion_family](trial)
    pipeline = build_champion(tree_preprocessor).set_params(**params)
    scores = cross_val_score(
        pipeline,
        X_train_sub,
        y_train_sub,
        cv=GroupKFold(n_splits=n_groups_sub),
        groups=years_sub,
        scoring="f1_macro",
        n_jobs=1,
    )
    return float(scores.mean())


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=40)
print(f"Best trial macro-F1 (CV, subsample): {study.best_value:.3f}")
print(f"Best params: {study.best_params}")

# %% [markdown]
# ## 8 — Refit on full training data and evaluate on test-2024 exactly once
#
# The tuned configuration is refit on the **full** 2016–2022 training set
# (not the subsample used for tuning), then evaluated on the 2024 test
# split — the single time this notebook touches the test set.

# %%
best_params = {f"classify__{k}": v for k, v in study.best_params.items()}
final_pipeline = build_champion(tree_preprocessor).set_params(**best_params)
final_pipeline.fit(X_train, y_train)

test_preds = final_pipeline.predict(X_test)
final_metrics = evaluate_predictions(y_test, test_preds)
passes = meets_acceptance_criteria(final_metrics)

print("=== FINAL TEST-2024 EVALUATION ===")
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
    "champion_family": champion_family,
    "winning_strategy": winning_strategy_row["model"],
    "tuned_hyperparameters": study.best_params,
    "test_2024_metrics": final_metrics,
    "acceptance_gate_passed": passes,
    "provenance": provenance,
}
card_path = PROCESSED_DIR / "a3_model_card.json"
card_path.write_text(json.dumps(model_card, indent=2))
print(f"Saved: {model_path}")
print(f"Saved: {card_path}")

# %% [markdown]
# ## 10 — A³ summary and A³-to-C handoff
#
# ### Model-comparison table (central portfolio artefact)
#
# See §5 and §6 above — 8 baseline/tree configurations + 4 imbalance
# strategies on the champion + 1 tuned final configuration, all scored on
# the 2023 validation split, with exactly one test-2024 evaluation.
#
# ### A³-phase acceptance checklist
#
# ```text
# [ ] U-phase §10 preprocessing implemented as a single sklearn Pipeline
# [ ] Baselines (random guess, majority class, logistic regression) scored
# [ ] 4 tree families x {default, class-weighted} scored on validation
# [ ] Champion selected by validation macro-F1
# [ ] SMOTE / ADASYN / threshold moving / ordinal classification compared
#     on the champion only
# [ ] Winning (model, strategy) combination tuned with Optuna (<= 40 trials)
# [ ] Tuned configuration refit on the FULL 2016-2022 training set
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

- [ ] **Step 2: Materialize and execute the full notebook**

Run:
```bash
uv run jupytext --to notebook notebooks/03_A3_Phase.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_A3_Phase.ipynb
```
Expected: exits 0, no exception cells, and `data/processed/a3_best_model.joblib` + `data/processed/a3_model_card.json` exist afterward. Verify the CV fold count printed by Optuna's trial logs (or add a one-line `print(n_groups_sub)`) equals the number of distinct years actually present in `years_sub` — this confirms `objective()`'s `GroupKFold` is grouping by real accident years, not an arbitrary index-derived value.

- [ ] **Step 3: Commit**

```bash
uv run jupytext --sync notebooks/03_A3_Phase.ipynb
git add notebooks/03_A3_Phase.ipynb notebooks/03_A3_Phase.py data/processed/a3_best_model.joblib data/processed/a3_model_card.json
git commit -m "feat: A3 notebook part 2 - imbalance comparison, tuning, final test evaluation"
```

---

### Task 11: Documentation updates and final verification

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/GLOSSARY.md`

**Interfaces:**
- Consumes: `champion_family`, `winning_strategy_row`, `final_metrics`, `passes` from the executed Task 10 notebook (read the printed output / `a3_model_card.json`, do not re-derive).

- [ ] **Step 1: Update `AGENTS.md`'s notebook status table**

In `AGENTS.md`, within the `<!-- AUTO-MANAGED: architecture -->` block, change:

```
│   ├── 03_A3_Phase.ipynb   # Modelling & tuning (TODO)
```

to:

```
│   ├── 03_A3_Phase.ipynb   # Modelling & tuning (done)
```

and add the two new library files to the same block's file listing, next to the existing `features/` and `models/` entries:

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
A hyperparameter-optimisation library using sequential model-based search (TPE — Tree-structured Parzen Estimator — by default). Used in A³ to tune only the single winning (model, imbalance-strategy) combination, bounded to a fixed trial count to respect the project's single-workstation compute budget.

**GroupKFold**
A cross-validation splitter that guarantees all rows sharing a group value (here: `UJAHR`, the accident year) fall in the same fold. Used instead of a random `StratifiedKFold` to prevent a model from training and validating on the same year, which would leak temporal structure during model selection.

**Champion model**
The single (model, configuration) combination selected by highest validation macro-F1 among the Stufe-0/Stufe-1 candidates in A³. Only the champion is carried forward into the imbalance-strategy comparison and hyperparameter tuning — not every trained configuration.
```

- [ ] **Step 3: Run the full test suite once more and lint**

Run: `uv run pytest -v && uv run ruff check . && uv run black --check .`
Expected: all tests pass; ruff and black report no violations. If black reports formatting differences, run `uv run black .` and re-check.

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md "docs/GLOSSARY.md"
git commit -m "docs: mark A3 phase complete, document new terms and architecture"
```

---

## Self-Review Notes

- **Spec coverage:** U-phase §10 table (cyclic/one-hot/target-encoding/log/scaling/drops) → Task 3; §10 imbalance menu (class weights/SMOTE/ADASYN/threshold moving/ordinal) → Tasks 5,6,7,8 + notebook §4/§6; §10 CV hint (GroupKFold/TimeSeriesSplit, not random) → Task 9 §2 and Task 10 §7; Q-phase acceptance criteria (macro-F1 ≥0.55, recall(1) ≥0.50) → Task 4 + notebook §8; Q-phase reproducibility/interpretability/model-size constraints → Global Constraints + notebook §0/§9; new prompt-file + disclosure requirement → Task 1.
- **Placeholder scan:** no "TBD"/"handle edge cases" left; `objective()`'s `groups=` now uses the real `years_sub` array (carried alongside the §6 subsample) instead of an index-derived placeholder — fixed during self-review, not left for the implementer to discover.
- **Type consistency:** `build_preprocessor(scale_for_linear: bool)` used identically in Tasks 3, 5, 6, 9; `evaluate_predictions(y_true, y_pred) -> dict` keys (`macro_f1`, `recall_class_1/2/3`, `confusion_matrix`) match across Task 4, 9, 10; `OrdinalClassifier.classes_`/`.predict_proba` used consistently in Task 7 and Task 10 §6d.

````
## Follow Up Prompt

````md
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

````

## A³ Champion Pivot — Recall-Aware Candidate Selection

**Claude Code (Sonnet 5), effort: medium (Anthropic, 2026):**

### Initial prompt

Not a single verbatim user message like the entry above — this plan was
produced in-conversation with the `superpowers:brainstorming` skill,
followed by `superpowers:writing-plans`, picking up directly after the
original A³ modelling plan's live execution (Task 10, `notebooks/03_A3_Phase.ipynb`
§5) had already run and produced a result: the original "champion =
highest validation macro-F1, alone" rule had selected
`random_forest_balanced` (macro-F1 0.410), but that model's
recall(class 1) was only 0.229 — far below the Q-phase's own ≥0.50
acceptance-gate threshold — while the untuned `catboost_balanced` (0.363
macro-F1, 0.537 recall) and `lightgbm_balanced` (0.364 macro-F1, 0.588
recall) already cleared that harder gate. The user and Claude Code
discussed this discrepancy against `docs/project/PROJEKTPLAN_SETUP.md`'s
literature-derived roadmap, which independently predicted Random Forest
would cap around 0.50–0.55 macro-F1 while CatBoost/LightGBM had a much
higher ceiling — concluding that continuing to tune Random Forest was very
unlikely to ever satisfy the recall gate, and that the selection rule
itself needed to change before Task 10/11 (tuning, final evaluation,
documentation) could sensibly proceed.

The resulting task-by-task plan lives at
`docs/superpowers/plans/2026-07-06-a3-champion-pivot.md`, implemented with
the `superpowers:subagent-driven-development` skill. (An early draft of
this same plan's content, produced before the final task-by-task file was
written, is also preserved verbatim above under "Follow Up Prompt" earlier
in this document.) What follows is a summary of the plan's Goal,
Architecture, and Global Constraints, plus a task-by-task outline — see the
plan file for full code-level detail.

**Goal:** Finish the unfinished A³-phase work (original plan's Task 10/11)
by replacing its "champion = highest validation macro-F1" selection rule
with a recall-aware rule, carrying **CatBoost and LightGBM** forward as the
two candidates (not Random Forest), tuning and evaluating them, and
completing the still-pending documentation task — while touching the 2024
test set exactly once.

**Architecture:** Notebook-orchestration changes only — no new files.
`notebooks/03_A3_Phase.ipynb`'s §5–§10 are rewritten to run the existing §6
imbalance-strategy menu (SMOTE/ADASYN/threshold-moving/ordinal, already
implemented in `src/unfallatlas/models/imbalance.py` and `ordinal.py`) over
**two** candidate families instead of one, select within each family using
a new shared helper, tune each family's winner separately with Optuna
(still CV-only, never touching test), pick one final cross-family winner
using the same helper, then refit once on full training data and evaluate
once on test-2024. The new selection rule is added as a small,
independently-testable pure function in `src/unfallatlas/models/evaluate.py`
rather than inline notebook code, since it is called three times.

**Global Constraints (summary; full list, including the ⚠️ AMENDED champion-
selection rule and the checkpointing/Optuna-study-naming rules, is in the
plan file):**
- All Q-phase/U-phase global constraints from the original A³ plan remain
  fully binding (target column, primary/secondary metrics, chronological
  split, `GroupKFold` by year, U-phase §10 preprocessing unchanged, compute
  budget, model-size, interpretability, reproducibility).
- ⚠️ **AMENDED — champion selection.** Only `catboost_balanced` and
  `lightgbm_balanced` advance past §5. Within and across those two
  families, the winner is chosen by `select_best_candidate()`: highest
  macro-F1 among candidates that clear recall(class 1) ≥ 0.50; if none
  clear it, highest `(macro_f1 + recall_class_1) / 2`.
- Compute budget: two families now share the existing Optuna trial budget
  (9 trials per family, 18 total, same as the prior single-family budget of 40).
- Exactly one test-2024 evaluation, ever — the single most important
  constraint for this plan specifically, since it now compares two tuned
  candidates before the one allowed test touch.
- Each family gets its own Optuna `study_name` to avoid a resumed study
  silently mixing an old, incompatible search space (a bug class already
  hit once in an earlier session).

**Tasks (see the plan file for full code/tests):**
1. `select_best_candidate(rows, recall_threshold=RECALL_CLASS_1_THRESHOLD) -> pd.Series`
   in `src/unfallatlas/models/evaluate.py` — the shared, gate-aware
   selection rule, unit-tested for the "recall passers exist," "fallback to
   combined score," and "custom threshold" cases.
2. Notebook §5 rewritten to carry forward two candidates
   (`candidate_families`, `candidate_names`, `candidate_pipelines`) instead
   of a single champion.
3. Notebook §6 rewritten to run the imbalance-strategy comparison
   (SMOTE/ADASYN/threshold-moving/ordinal) over both candidate families in
   a loop, producing `winning_strategy_per_family`.
4. Notebook §7 rewritten to tune each family separately with Optuna (9
   trials each), tracking recall(class 1) per trial via
   `trial.set_user_attr` and selecting the best trial per family with
   `select_best_candidate`, producing `tuned_candidates`.
5. Notebook §8 rewritten: final cross-family selection via
   `select_best_candidate` over both families' tuned CV scores, single
   refit on the full 2016–2022 training set, single test-2024 evaluation —
   the only place `X_test`/`y_test` may be referenced in the whole notebook.
6. Execute the full notebook end-to-end (as a detached process — the run
   is long) and verify: per-family Optuna progress streams, exactly one
   `FINAL TEST-2024` log line, a valid model card with exactly two
   per-family tuned entries, model artifact under the 500 MB soft limit.
7. Finish the original plan's still-pending Task 11 — `AGENTS.md` notebook
   status, `docs/GLOSSARY.md` new terms (Threshold Moving, Ordinal
   Classification, Optuna, GroupKFold, Champion candidates), full test
   suite + lint.

---

## A³ OSM Feature Integration + Deferred Review Findings

**Claude Code (Sonnet 5), effort: medium (Anthropic, 2026):**

### Initial prompt

Not a single verbatim user message — this plan was produced with the
`superpowers:writing-plans` skill, picking up after two prerequisite pieces
of work had both landed: the U-phase OSM/H3 road-context feature addendum
(`docs/prompts/02_prompts_phase_u.md`'s "U-Phase Addendum — OSM/H3
Road-Context Features" entry, plan file
`docs/superpowers/plans/2026-07-06-u-phase-osm-spatial-features.md`), which
had deliberately scoped out wiring the new columns into A³ ("consuming the
new columns into A3's `build_preprocessor` ... is a separate, following
plan"); and the A³ champion-pivot plan immediately above, whose final
whole-branch review had deferred 3 Minor findings explicitly to "the next
plan (updating A3 to consume these new spatial features)." This plan is
that promised next plan, named in both places it was promised, closing out
both threads in one pass.

The resulting task-by-task plan lives at
`docs/superpowers/plans/2026-07-09-a3-osm-feature-integration.md`,
implemented with the `superpowers:subagent-driven-development` skill. What
follows is a summary of the plan's Goal, Architecture, and Global
Constraints, plus a task-by-task outline — see the plan file for full
code-level detail, and `docs/osm-feature-retrospective.md` for a fuller
narrative on the measured result.

**Goal:** Wire the 5 OSM road-context columns
(`osm_dominant_road_class`, `osm_maxspeed_mean`, `osm_maxspeed_max`,
`osm_road_density`, `osm_way_count`) — already fetched, aggregated, joined,
and specified in U-phase §10 — into A³'s `build_preprocessor()` so every
model actually trains on them, fold in the 3 Minor findings deferred from
the champion-pivot plan's final review, then re-run
`03_A3_Phase.ipynb` end-to-end with the enlarged feature set and record the
result.

**Architecture:** One library change
(`src/unfallatlas/features/preprocessing.py` — extend `build_preprocessor()`'s
existing `PLAIN_NUMERIC_COLUMNS`/`LOG1P_COLUMNS` lists and add one small new
pipeline branch for the categorical `osm_dominant_road_class` column) plus
notebook-only changes to `notebooks/03_A3_Phase.ipynb` (three isolated,
already-diagnosed bugfixes + a live end-to-end re-run). No new files, no
new model families, no new hyperparameter search space, no change to the
acceptance gate or champion-selection rule.

**Global Constraints (summary; full list in the plan file):**
- All Q-phase/U-phase/champion-pivot global constraints remain binding —
  this plan changes inputs to the pipeline, not the acceptance rules or
  the selection rule.
- U decides, A³ implements: the missing-value strategy, transform, and
  scaling for every OSM column is already decided in U-phase §10 — this
  plan implements that table verbatim, it does not re-decide it.
- Exactly one test-2024 evaluation, ever — unchanged from the champion-pivot plan.
- Compute budget unchanged (`SUBSAMPLE_CAP = 500_000`, 9 Optuna trials per family).
- Checkpointing rule is the **opposite** of every prior plan's: do NOT copy
  any checkpoint forward — every single cached model was fit against the
  old `build_preprocessor()` output shape and is invalid the moment Task 1 lands.

**Tasks (see the plan file for full code/tests):**
1. Wire the 5 OSM columns into `build_preprocessor()`: extend
   `LOG1P_COLUMNS`/`PLAIN_NUMERIC_COLUMNS` (DRY reuse of existing pipeline
   branches for the 4 numeric columns) and add a new
   `osm_categorical_pipeline` (mode/"unknown"-impute + one-hot) for
   `osm_dominant_road_class`. 4 new unit tests covering one-hot encoding,
   missing-value handling for the categorical and numeric OSM columns, and
   confirming `h3_cell`/`dwd_station_id` stay silently dropped.
2. Fold in the 3 deferred champion-pivot review findings: tighten the §6
   `strategy_rows` filter to an explicit allow-list (was a `startswith`
   prefix match that incidentally also matched `{family}_default` rows with
   no refit path); document the §2 CV cell as an intentional, unconsumed
   sanity check rather than latent dead code; persist the full
   Stufe-0/1/§6 model-comparison table to a committed
   `data/processed/a3_model_comparison.csv` (previously only summarized in
   `a3_model_card.json`, with the full table unrecoverable from git history
   since `nbstripout` strips notebook outputs and the progress log is
   git-ignored).
3. Execute the full A³ notebook end-to-end with OSM features (detached
   process, no checkpoints copied forward) and record the result against
   the pre-OSM baseline.
4. Documentation: a new `docs/AI TOOL DISCLOSURE.md` row.

**Result (executed on this branch; see `docs/osm-feature-retrospective.md`
for the fuller narrative):** champion family = `lightgbm` (tuned) both
before and after. Test-2024 macro-F1 moved 0.358 → 0.362 (+0.004);
recall(class 1) moved 0.615 → 0.649 (+0.034). Q-phase acceptance gate
(macro-F1 ≥ 0.55 **and** recall(1) ≥ 0.50) remained **FAIL** before and
after — both metrics improved, but not by enough to flip the gate outcome;
macro-F1 is still roughly 0.19 short of the 0.55 threshold. The
retrospective's Optuna-trial analysis suggests this is closer to a
precision/recall tradeoff-frontier plateau for this feature-set/model-family
combination than an under-searched hyperparameter space, and lists several
not-yet-explored, higher-leverage directions (per-fold-safe resampling
inside Optuna's CV, a deliberately-tuned threshold-moving step, additional
feature sources, revisiting the ordinal decomposition on the enlarged
feature set) for future work.