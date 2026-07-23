# Presentation and Notebook Refactor Design

**Date:** 2026-07-23  
**Status:** Approved for implementation  
**Scope:** Q, U, A³, and C notebooks; presentation exporter; generated HTML; reusable
visualization and C-phase analysis code  
**Language:** English throughout

## 1. Objective

Refactor the complete QUA³CK report into a coherent, presentation-ready analytical
narrative. The work must resolve notebook/HTML rendering differences, standardize charts
on interactive Plotly output, replace avoidable raw value dumps with clearer visual
evidence, integrate the multiclass-to-binary target decision into the main story, and
finalize the C phase using the available persisted candidate models.

The refactor must preserve the scientific record:

- intentional console output remains visible when it is evidence;
- the original three-class investigation is not rewritten out of history;
- validation remains the model-comparison and selection surface;
- Test-2024 remains reserved for the selected champion;
- existing uncommitted notebook outputs, models, and exports are preserved and built upon;
- no commits are created during implementation.

## 2. Narrative Architecture

### 2.1 Q phase

Q retains the original three-class research question. The later operational target
revision is integrated into the target-definition and success-metric sections instead of
appearing as an appended addendum.

The phase distinguishes:

1. the original scientific question about three severity levels;
2. the feasibility gate used to decide whether that target is supportable;
3. the evidence-driven KSI-versus-slight operational reformulation justified in A³.

The current numbering gap and appended gate-revision section are removed. The summary
closes the phase once, after the revised target policy has been explained.

### 2.2 U phase

U remains an exploratory investigation of the original severity target. Its strongest
feasibility evidence is organized into a continuous target-viability thread:

- extreme class imbalance;
- weak feature-to-target Cramér's V values;
- nearly invariant severity shares across important categories;
- missing physical determinants of injury outcome.

U introduces binary KSI as a fallback framing without claiming modelling results that
belong in A³. The preprocessing contract remains target-independent where possible.
Duplicate interpretations, stale cross-references, and the development acceptance
checklist are removed.

### 2.3 A³ phase

A³ becomes a two-part decision funnel.

#### Part I — Three-class feasibility study

1. Reproducible setup, data contract, chronological split, and evaluation protocol.
2. Baselines and model-family comparison.
3. Imbalance strategies and tuning.
4. Validation selection and the three-class result.
5. Empirical and arithmetic ceiling analysis.
6. Explicit target decision.

The current section 10 summary/checklist no longer interrupts the phase before the target
decision. The multiclass work is presented as necessary feasibility evidence rather than
as a completed report followed by a binary appendix.

#### Part II — Binary KSI modelling study

1. Binary target and acceptance criteria.
2. Stage 0 baselines and Stage 1 candidate search.
3. Gate-aware validation selection.
4. Winning-family tuning and calibration refinement.
5. Validation operating-point selection.
6. Full refit and one final Test-2024 evaluation.
7. Binary-target evidence, artifacts, and handoff to C.
8. One final A³ summary.

The negative result from multi-objective tuning/calibration remains, but is placed beside
the winning-family tuning decision rather than appended after the final test result.

### 2.4 C phase

C uses persisted candidate pipelines at three levels:

- **All ten candidates:** interactive validation comparison table and macro-F1 versus
  Recall(KSI) plot, including baselines for auditability.
- **Substantive score-producing models:** Val-2023 ROC and precision-recall comparison,
  with no-skill references where meaningful.
- **Serious tree finalists:** Random Forest, XGBoost, LightGBM, and CatBoost for measured
  latency, missing-feature robustness, prediction disagreement, and model-agnostic
  permutation importance on a fixed validation sample.

The selected Random Forest retains the threshold-specific Test-2024 confirmation, error
slices, confusion matrix, and detailed SHAP analysis. Other candidates are not evaluated
retrospectively on Test-2024.

The qualitative comparison uses measured values. It must not copy champion latency into
runner-up rows, invent per-family training costs, or rank candidates using unsupported
criteria. Final conclusions are regenerated from the measured analysis.

## 3. Visualization Contract

### 3.1 Plotly-only charts

All chart-producing reusable helpers and notebook cells return Plotly figures. The
remaining Matplotlib implementation in `src/unfallatlas/viz/metrics_viz.py` is replaced,
including the helpers used in A³.11 and A³.19.

Notebook output uses the saved `application/vnd.plotly.v1+json` MIME representation. The
presentation exporter loads the locally vendored Plotly runtime so exported figures remain
interactive and work offline.

### 3.2 Tables and textual evidence

Presentation-critical comparisons use Plotly tables, heatmaps, or indicator panels where
that improves comprehension. Examples include:

- model comparisons;
- gate results;
- confusion matrices;
- feature rankings;
- artifact and provenance summaries.

Large data-audit tables may remain semantic HTML tables when tabular exploration is the
clearest form.

Console output is not globally removed, hidden, or suppressed. If a notebook intentionally
prints evidence, the exporter displays it. Raw numerical dumps are replaced only when a
visualization communicates the same evidence more clearly. Training diagnostics may
continue to use the existing progress log and concise visible status messages.

### 3.3 Common styling

All figures use shared English labels, color tokens, typography, hover formatting, and
layout defaults. Severity and KSI colors remain semantically stable across phases. Figure
titles describe what is plotted; interpretation remains in surrounding narrative or
subtitles.

## 4. HTML and Markdown Rendering

The final A³ Markdown table fails because it is indented inside a list item. The Jupyter
renderer accepts this extended nested-table form, while the nbconvert Markdown path emits
the pipe syntax as paragraph text.

The confusion matrix is replaced with an interactive Plotly heatmap. Other Markdown
structures are audited for renderer differences, including:

- nested tables;
- task lists;
- raw HTML;
- blockquotes;
- heading anchors and cross-references;
- local images and resources;
- unsupported widget-only outputs.

Regression checks fail if exported HTML contains literal table pipe syntax inside rendered
paragraphs, missing Plotly containers/runtime, static chart MIME output, malformed local
resources, or broken anchors.

The German label `Arbeitsbaum` is deliberate Git terminology, but it is not
audience-friendly. Repository, commit, branch, and modified-state information moves into a
collapsed English technical-provenance section. The visible header contains only
audience-relevant document and export status.

## 5. Candidate Artifact Registry

A³ records an explicit registry for persisted binary candidate artifacts. Each entry
contains:

- model/family name;
- relative artifact path;
- training sample size;
- score interface (`predict_proba`, `decision_function`, or prediction-only);
- evaluation role;
- artifact fingerprint;
- data/model-card fingerprint required for downstream cache validation.

C loads from this registry instead of deriving checkpoint paths from assumptions. Missing,
stale, or incompatible artifacts produce a clear error naming the affected candidate.

## 6. C-Phase Analysis Flow

Candidate models are loaded sequentially to bound memory use:

1. Load one candidate.
2. Recompute or verify Val-2023 scores and metrics.
3. Measure warmed-up repeated inference latency on the same fixed sample.
4. Run the same missing-feature robustness probe.
5. Persist compact derived evidence.
6. Release the model before loading the next candidate.

For the four tree finalists, C additionally calculates:

- pairwise prediction disagreement;
- model-agnostic permutation importance on one fixed validation sample;
- feature-rank agreement and divergence.

The champion alone receives detailed Test-2024 and SHAP analysis.

Derived C evidence is cached with artifact and data fingerprints. A cache mismatch forces
recomputation rather than silently reusing stale results.

## 7. Execution and Reproducibility

Expensive A³ stages use explicit load-or-fit behavior. Existing checkpoints and current
uncommitted artifacts are preserved.

Optuna studies resume only for the number of missing trials. Re-executing a notebook must
not silently add another full trial budget to an already-complete study.

Execution order:

1. Edit `.ipynb` notebooks and reusable library/exporter code.
2. Regenerate Jupytext mirrors.
3. Run focused unit and exporter regression tests.
4. Execute U, A³, and C using caches/checkpoints; Q requires no computational execution.
5. Export all phases from saved notebook outputs.
6. Run complete tests and linting.
7. Run structural HTML validation and Playwright browser checks.
8. Perform a final editorial review of the rendered HTML.

No implementation commit is created.

## 8. Verification Requirements

The refactor is complete only when all of the following hold:

- Q, U, A³, and C form one continuous English narrative.
- Multiclass feasibility naturally motivates the binary KSI study.
- No stale addenda, checklists, placeholder language, or obsolete artifact limitations
  remain.
- All charts are Plotly figures.
- A³.11 and A³.19 are interactive in exported HTML.
- Every exported Plotly figure responds to browser interaction.
- No chart is exported only as a static PNG.
- Markdown tables and other structures render consistently between notebooks and HTML.
- Necessary console evidence remains visible.
- Avoidable raw numerical dumps are replaced with clearer presentation forms.
- C uses all candidates where meaningful and measured finalist comparisons where valid.
- Non-champion models do not influence decisions through Test-2024.
- The C inference contract matches the confirmed champion and threshold.
- Jupytext mirrors match their `.ipynb` sources.
- Targeted and full pytest suites pass.
- Ruff passes.
- Export validation and Playwright browser checks pass without browser console errors.

## 9. Non-Goals

- Building the Streamlit K phase.
- Creating German presenter notes.
- Producing a bilingual report or glossary.
- Retraining models merely to replace valid persisted artifacts.
- Rewriting the scientific history to imply that the binary target was the original target.
- Committing notebook or generated-output changes during this implementation.
