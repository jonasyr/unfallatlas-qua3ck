# C-Phase Notebook ("Conclude & Compare") — Design

**Status:** Approved by user, ready for `writing-plans`.

## Context

`notebooks/04_C_Phase.ipynb`/`.py` currently exists only as an empty stub
(`# QUA³CK — 04_C Phase (TODO)`). The A³ phase (`notebooks/03_A3_Phase.py`)
is complete and, after the binary-KSI reformulation (§12–§21), produced a
final champion:

- **Champion**: `random_forest`, class-weighted/balanced, Optuna-tuned
  (`n_estimators=180, max_depth=23, min_samples_leaf=8`), gate-optimal
  decision threshold **0.4986**.
- **Test-2024** (evaluated exactly once): macro-F1 **0.6026**,
  recall(KSI) **0.5255**, recall(slight) 0.7615. Both acceptance gates
  (macro-F1 ≥ 0.55, recall(KSI) ≥ 0.50) **PASS**.
- **Runner-ups** (Val-2023): `xgboost` (macro-F1 0.5699, recall_ksi 0.6824),
  `lightgbm` (macro-F1 0.5662, recall_ksi 0.6897) — both have higher
  recall(KSI) than the champion but lower macro-F1.
- Full 10-candidate comparison in `data/processed/a3_binary_model_comparison.csv`;
  model card in `data/processed/a3_binary_model_card.json`; fitted pipeline in
  `data/processed/a3_binary_best_model.joblib`.
- A³ §20 already computed Cramér's V feature-association evidence
  (`UART` strongest at 0.1801) and the champion's built-in feature
  importances (lean on OSM road-context + DWD weather/geo features).
- A³ §11 contains the arithmetic/empirical "3-class ceiling" proof
  motivating the pivot from multiclass `UKATGEORIE` to binary KSI.
- A³ §10's transition note (written pre-pivot) is partially stale — it
  names only SHAP + literature benchmark + limitations as C-phase scope.
  The original `docs/project/PROJEKTPLAN_SETUP.md` QUA³CK-Phasen-Mapping
  section specifies a broader scope (comparison table, ROC/PR, confusion
  matrices, imbalance-strategy comparison, SHAP, literature alignment,
  limitations) — **this design follows the broader PROJEKTPLAN scope**,
  which supersedes the stale A³ §10 note.
- Reference structural pattern from `Degrees-of-No-Return-App/notebooks/C-Phase.ipynb`
  (different domain — regression/sea-level modelling): intro → systematic
  model comparison → domain-appropriate diagnostics → formal KPI Go/No-Go →
  qualitative weighted evaluation matrix → domain-plausibility check → final
  decision → K-phase handoff/artifact export → summary. Adapted below for a
  binary classifier (their residual analysis and IPCC-plausibility check
  don't transfer directly).
- Repo convention: notebooks are jupytext percent-format `.py` paired with
  `.ipynb`, German prose / English code and identifiers, numbered `## N —`
  sections, explicit inter-phase handoff notes (see `03_A3_Phase.py` style).

## Goal

Produce a complete, self-contained `04_C_Phase.ipynb` (+ paired `.py`) that:

1. Consumes the A³ artifacts (no retraining) to present the full binary
   champion-search comparison, explainability, and evaluation story.
2. Delivers an honest, evidence-grounded synthesis of why `random_forest`
   is the confirmed champion despite runner-ups having higher recall(KSI).
3. Hands off a complete, self-contained artifact package to the K-phase
   Streamlit app (model, threshold, contract).

## Non-goals

- No retraining, no new hyperparameter search, no new feature engineering.
- No re-litigating the 3-class-vs-binary pivot (already settled in A³ §11–12)
  beyond a brief pointer.
- No independent re-derivation of the Cramér's V association analysis — C
  builds on A³ §20 rather than duplicating it (per user decision).
- No SMOTE/ADASYN re-comparison for the binary target — class-weighting +
  gate-optimal threshold moving is the adopted imbalance strategy; the
  historical multiclass SMOTE/ADASYN work (A³ §6) is referenced only as
  context for why it was superseded by the ceiling proof (A³ §11).

## Section plan

Mirrors `03_A3_Phase.py` conventions: jupytext percent format, numbered
`## N —` markdown headers, German prose / English code, explicit "Position
in QUA³CK" framing at the top and an explicit handoff note at the bottom.

**§0 — Setup, artifact loading, and position in QUA³CK**
Load `a3_binary_best_model.joblib`, `a3_binary_model_card.json`,
`a3_binary_model_comparison.csv`, and the same U-phase processed
Test-2024/Val-2023 splits A³ used (for recomputing predictions where needed,
e.g. SHAP, error slicing). State position in QUA³CK, restate the champion
and headline numbers, list section overview.

**§1 — Systematic model comparison**
Full 10-candidate table (`a3_binary_model_comparison.csv`) rendered as a
formatted DataFrame/table. ROC and PR curves for champion + 2 runner-ups
(xgboost, lightgbm) computed from Test-2024 predicted probabilities.
Confusion-matrix heatmaps for the champion (and optionally runner-ups).
Short note: binary framing's imbalance handling is class-weighting +
gate-optimal threshold moving (no SMOTE at ~20/80 imbalance); pointer to
A³ §11 for why the earlier multiclass SMOTE/ADASYN comparison (§6) was
superseded.

**§2 — Error-slice diagnostics**
Break down false negatives (missed KSI) and false positives on Test-2024 by
`UART` (accident type), OSM road-context buckets, weather features, and
time-of-day, to check whether errors cluster systematically rather than
randomly. Bar/heatmap visualizations of error rate by slice.

**§3 — Formal KPI Go/No-Go validation**
Explicit table restating Q-phase gate definitions (macro-F1 ≥ 0.55,
recall(KSI) ≥ 0.50) next to Val-2023 and Test-2024 champion numbers, with an
explicit PASS/FAIL verdict per gate and overall.

**§4 — Qualitative weighted evaluation matrix**
Champion vs. xgboost vs. lightgbm scored across: macro-F1, recall(KSI),
inference latency (measured, not assumed — time a batch prediction),
interpretability (tree count/depth as a proxy, native feature-importance
availability), robustness to missing OSM/DWD features (does the pipeline
degrade gracefully or error out — check imputation strategy from U-phase
contract), training cost (wall-clock from A³ logs / n_train). Weighted
scoring table with an explicit weight-choice justification (recall(KSI) and
macro-F1 weighted highest, consistent with the Q-phase gate priorities).

**§5 — SHAP explainability**
`shap.TreeExplainer` on the champion pipeline, run on a stratified ~5,000-row
sample of Test-2024 (full 1.5M rows is not computationally practical — note
this explicitly as a documented sampling choice, not a silent shortcut).
Global: summary/beeswarm plot + mean-|SHAP| importance bar plot. Local:
waterfall/force plots for 3–4 concrete cases — a true-positive KSI, a
false-negative KSI, a false-positive slight, and a true-negative — each with
a short narrative interpretation.

**§6 — Literature alignment**
Builds on A³ §20 (Cramér's V, native feature importances) rather than
recomputing association from scratch. Adds the SHAP-based importance view
alongside the native one. Compares Test-2024 macro-F1 (0.6026) against the
Q-phase literature anchors (Santos 2022 ~0.60, Pakgohar 2021 ~0.62,
Schlößler 2024 ~0.65) and discusses whether the SHAP top features
(road-context/weather-leaning) are consistent with what the literature
identifies as dominant KSI predictors.

**§7 — Limitations**
Honest discussion: selection bias (police-recorded accidents only），missing
demographic/physical-injury determinants (seatbelt use, occupant age,
impact speed — noted in A³ as living in restricted Destatis microdata),
correlation ≠ causation, geographic/temporal coverage limits, residual
class imbalance effects on recall(KSI), sensitivity of results to the
chosen decision threshold (brief sensitivity note referencing §3).

**§8 — Final model decision / synthesis**
Explicit synthesis paragraph + decision table tying together §3 (KPI gate),
§4 (qualitative matrix), §5–6 (SHAP/literature), confirming `random_forest`
as champion with reasoning for why higher-recall(KSI) runner-ups were not
promoted (macro-F1 trade-off, per the qualitative matrix weighting).

**§9 — Handoff to K-phase: artifact export**
Full package: confirm/re-save pointer to `a3_binary_best_model.joblib` and
threshold 0.4986 (no re-save needed if unchanged — verify hash/metadata
matches A³'s saved model card); a new **inference contract** cell/table
listing every required input column, dtype, valid range/categories, and
source (U-phase feature vs. DWD/OSM enrichment), so the K-phase Streamlit
app has everything it needs without re-reading the earlier notebooks.

**§10 — Summary**
What was achieved (bullet recap of §1–9), outlook to K-phase, limitations
restated briefly (pointer to §7, not a re-derivation).

## Data flow

```
data/processed/a3_binary_best_model.joblib  ──┐
data/processed/a3_binary_model_card.json    ──┼──> 04_C_Phase.ipynb ──> reports/figures/c_phase_*.png
data/processed/a3_binary_model_comparison.csv─┤                    └──> (confirmed) inference contract
U-phase processed Test-2024/Val-2023 splits ──┘                         for K-phase Streamlit app
```

No new files are written to `data/processed/` unless the inference-contract
cell is also persisted as a small JSON/YAML artifact for the K-phase app to
import directly (recommended — avoids the K-phase implementer re-deriving
it from notebook prose). New figures go to `reports/figures/`.

## Testing / verification

- Notebook must execute end-to-end top-to-bottom without errors
  (`jupytext --sync` + full re-execution), consistent with the project's
  existing "real end-to-end notebook execution" standard (see A³ SVM
  algorithm-selection plan).
- All numbers quoted in markdown prose must come from actually-executed
  cells in this notebook or direct reads of the cited A³ artifacts — no
  fabricated/estimated figures.
- SHAP sample size and any other sampling choices must be stated explicitly
  in the notebook text, not silently applied.
- Confirm the champion pipeline reloaded from the joblib reproduces the
  Test-2024 macro-F1/recall(KSI) numbers already on record (sanity check
  before building anything on top of it).

## Resolved decisions

- **Plotting library**: `plotly.express`/`plotly.io`, matching A³'s own
  import convention (`03_A3_Phase.py` §0), for all custom charts (ROC/PR,
  confusion matrices, error-slice bars, qualitative-matrix radar/bar). SHAP's
  own plotting functions (`shap.summary_plot`, waterfall/force plots) are
  matplotlib-based by construction (shap library default) — this is an
  accepted exception, not a convention break, since reimplementing SHAP's
  plots in plotly would add risk for no benefit.
- **Inference contract persistence**: saved as
  `data/processed/c_phase_inference_contract.json`, matching the
  `a3_binary_model_card.json` convention, so the K-phase Streamlit app can
  import it directly instead of re-deriving it from notebook prose.
