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
# NOTE: build_xgboost_pipeline/build_lightgbm_pipeline/build_catboost_pipeline
# all accept use_gpu: bool | None = None (auto-detects via gpu_available(),
# added after this plan was first written — see notebook Part 1's USE_GPU /
# _use_gpu_resolved cell). Thread _use_gpu_resolved through here explicitly:
# this is exactly where GPU acceleration matters most in this notebook — the
# Optuna loop below refits the champion dozens of times (n_trials x CV folds),
# not just once like every earlier section.
champion_builder = {
    "random_forest": lambda pre, **kw: build_random_forest_pipeline(pre, **kw),
    "xgboost": lambda pre, **kw: build_xgboost_pipeline(pre, use_gpu=_use_gpu_resolved, **kw),
    "lightgbm": lambda pre, **kw: build_lightgbm_pipeline(pre, use_gpu=_use_gpu_resolved, **kw),
    "catboost": lambda pre, **kw: build_catboost_pipeline(pre, use_gpu=_use_gpu_resolved, **kw),
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
#
# **GPU reminder.** `objective()` below calls `build_champion(tree_preprocessor)`
# once per trial per fold (`n_trials x n_groups_sub` fits total) — by far the
# most fit-heavy loop in this notebook. `champion_builder` (§6, just above)
# already threads `_use_gpu_resolved` into every XGBoost/LightGBM/CatBoost
# call, so this loop runs on GPU automatically whenever `USE_GPU` resolves to
# `True` — do not silently drop that kwarg when adapting this cell.

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
