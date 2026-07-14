# Binary KSI Front Refinement & Evidence — Design

**Status:** Approved by user, ready for `writing-plans`.

## Context

The A³ binary KSI champion search (see
`docs/superpowers/plans/2026-07-14-svm-algorithm-selection.md`) produced a
random_forest champion with Test-2024 macro-F1 = 0.6026, recall(KSI) =
0.5255, clearing the acceptance gate (macro-F1 ≥ 0.55 and recall(KSI) ≥
0.50).

The user asked whether any technique could push this significantly higher
(their reference point: a friend's unrelated project scoring ~0.9x). This
was investigated (via `superpowers:brainstorming`) against the project's own
prior evidence, not assumption:

- `docs/project/Technical_Review_Next_Steps.md` already established, over 19
  three-class configurations, an empirical ceiling of macro-F1 = 0.424 and a
  feature-association ceiling of Cramér's V ≤ 0.13 (against the *3-class*
  target) for the strongest available feature. The physical determinants of
  injury severity (impact speed, occupant age, seatbelt use, vehicle mass)
  are not present in the public Unfallatlas dataset — they live in
  access-restricted Destatis person/vehicle microdata.
- The binary reformulation's own pre-registered estimate (in the same
  document, category 4.1) was "realistically 0.58–0.65" macro-F1 — the
  achieved 0.6026 matches that estimate and matches published literature for
  comparable KSI-vs-slight tasks (Santos 2022 ~0.60, Pakgohar 2021 ~0.62,
  Schlößler 2024 ~0.65).
- Two concrete, previously-untried, low-risk levers remain, both cited by the
  technical review's own roadmap (categories 1.4 and 2.3) but never
  implemented for the binary model: multi-objective (Pareto) hyperparameter
  tuning that optimizes macro-F1 and recall(KSI) jointly instead of
  macro-F1 alone, and probability calibration before threshold search.

Conclusion reached and accepted by the user: no technique will move this
result into the 0.9x range on this feature set — that is empirically and
arithmetically foreclosed, not a tooling gap. What's left is (a) a bounded,
evidence-backed attempt at the two untried small levers, and (b) making the
"why 0.60 is the honest, defensible number" argument explicit for the binary
model the same way it already exists for the 3-class ceiling.

## Goal

1. Attempt multi-objective Optuna tuning + probability calibration for the
   binary champion family. Promote to champion only if it strictly
   dominates the current Val-2023 operating point on both macro-F1 and
   recall(KSI); otherwise report the negative result honestly and keep the
   current champion.
2. Add a binary-target-specific evidence section (fresh Cramér's V against
   the actual binary KSI label, feature importances of the actual binary
   champion, literature comparison) so the notebook's binary result is
   contextualized the same rigorous way the 3-class ceiling already is.

## Non-goals

- Re-running the full Stage 0/1 champion search across all ten families —
  scope is tuning refinement of the *already-selected* winning family only,
  matching the existing `## 16` section's own scope restriction.
- Any new feature engineering (interaction terms, exposure data, OSM
  geometry derivatives) — the technical review rates these high-effort,
  low-plausibility for this specific bottleneck (precision on a feature set
  with Cramér's V ≤ 0.13), and the user did not select that option.
- Touching the 3-class notebook path, the 3-class artifacts, or any file
  outside `notebooks/03_A3_Phase.ipynb`/`.py` and the three binary artifact
  files.
- Modifying `select_best_candidate`, `find_best_binary_threshold`, or any
  other shared library function's existing signature/behavior — only new
  call sites are added.

## Architecture

### Part 1 — Multi-objective tuning + calibration

New notebook section inserted immediately after the existing `## 17 —
Binary Refit, Gate-Optimal Threshold & Test-2024 Evaluation` and before the
existing `## 18 — Binary Artifacts: Save Pipeline & Model Card`. All
subsequent section numbers shift by one (`## 18`→`## 19`, `## 19`→`## 20`,
etc.) — a mechanical renumber, not a new duplicate-numbering defect.

**Step 1 — Multi-objective Optuna search.** Reuses
`BINARY_PARAM_SPACES[binary_champion_family]`, `BINARY_CV_DATA`, and
`_binary_fit_kwargs` exactly as `## 16` does. Only the study construction
and objective function differ:

```python
def binary_multiobj_objective(trial):
    params = param_space_fn(trial)
    gkf = GroupKFold(n_splits=3)
    fold_f1, fold_recall = [], []
    for tr_idx, va_idx in gkf.split(X_cv, y_cv, groups=groups_cv):
        p = build_champion_fn()
        p.set_params(**params)
        fit_kwargs = _binary_fit_kwargs(binary_champion_family, y_cv.iloc[tr_idx])
        p.fit(X_cv.iloc[tr_idx], y_cv.iloc[tr_idx], **fit_kwargs)
        pred = p.predict(X_cv.iloc[va_idx])
        fold_f1.append(f1_score(y_cv.iloc[va_idx], pred, average="macro"))
        fold_recall.append(recall_score(y_cv.iloc[va_idx], pred, labels=[1], average="macro"))
    return float(np.mean(fold_f1)), float(np.mean(fold_recall))


study_binary_mo = optuna.create_study(
    directions=["maximize", "maximize"],
    sampler=optuna.samplers.TPESampler(seed=SEED),
    study_name=f"binary_{binary_champion_family}_multiobj",
)
study_binary_mo.optimize(binary_multiobj_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
```

**Step 2 — Gate-aware Pareto pick.** Build a small DataFrame from
`study_binary_mo.best_trials` (one row per Pareto-optimal trial: `params`,
`macro_f1` = `trial.values[0]`, `recall_ksi` = `trial.values[1]`), then
reuse the existing `select_best_candidate(rows, recall_col="recall_ksi")` —
no new selection logic. This is the same function every other candidate
table in this notebook already uses.

**Step 3 — Calibrated refit.** Refit the picked params on
`BINARY_STAGE1_DATA[binary_champion_family]` (full training scale, same as
`## 17`), then wrap:

```python
from sklearn.calibration import CalibratedClassifierCV

pipeline_binary_calibrated = CalibratedClassifierCV(
    pipeline_binary_multiobj, method="isotonic", cv=3
)
pipeline_binary_calibrated.fit(X_refit_mo, y_refit_mo, **refit_fit_kwargs_mo)
```

`cv=3` cross-fits internally on the training data only — no Val/Test touch
at this step.

**Step 4 — Val-2023 threshold search.** Exactly the existing
`find_best_binary_threshold` call, applied to
`pipeline_binary_calibrated.predict_proba(X_val_bin)[:, 1]`. Produces
`best_threshold_mo`, `best_val_metrics_mo`.

**Step 5 — Promotion gate.**

```python
promote = (
    best_val_metrics_mo["macro_f1"] >= best_f1_val
    and best_val_metrics_mo["recall_ksi"] >= best_val_metrics["recall_ksi"]
)
```

using the Val metrics already computed in `## 17` (`best_f1_val`,
`best_val_metrics`) as the baseline — comparing Val-to-Val, never mixing CV
and Val numbers.

- **If `promote` is True:** evaluate `pipeline_binary_calibrated` on
  Test-2024 exactly once (same pattern as `## 17`'s Test evaluation),
  reassign `pipeline_binary_final`, `best_threshold`, `best_params`,
  `metrics_binary_test`, `gate_passed` to the new values so `## 18`
  (artifact save) and the results-summary section downstream pick them up
  unchanged — no duplicated save/reporting logic. Print a one-line note:
  `"Multi-objective tuning + calibration improved the Val-2023 operating
  point ({old} -> {new}); promoted to champion."`
- **If `promote` is False:** leave `pipeline_binary_final` and friends
  untouched. Print: `"Multi-objective tuning + calibration did not improve
  on the Val-2023 front (macro_f1={mo_f1:.4f} vs {old_f1:.4f}, recall_ksi=
  {mo_recall:.4f} vs {old_recall:.4f}); keeping the single-objective
  champion."` The candidate's Val metrics are printed for the record; it is
  never evaluated on Test-2024.

**Model-card provenance.** Add one field to the binary model card JSON
(built in `## 18`, shifted): `"multiobjective_refinement"`: a dict with
`"attempted": true`, `"promoted": bool`, `"val_macro_f1"`,
`"val_recall_ksi"` for the multi-objective candidate, and
`"promotion_rule"` (the exact string of the comparison used) — so the
negative or positive result is part of the permanent record, not just
notebook prose.

### Part 2 — Binary evidence section

New subsection, placed immediately before the existing results-summary
section (post-renumber, wherever `## 19 — Results Summary: Binary KSI
Classification` lands). Three cells:

**Cell A — Fresh Cramér's V against the binary label.**

```python
from scipy.stats import chi2_contingency


def cramers_v(feature: pd.Series, target: pd.Series) -> float:
    ct = pd.crosstab(feature, target)
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


binary_association_cols = ["Unfallart", "Unfalltyp", "Lichtverhaeltnisse", "Strassenzustand"]
binary_cramers_v = {
    col: cramers_v(df_full[col], y_binary_ksi_full)
    for col in binary_association_cols
    if col in df_full.columns
}
for col, v in sorted(binary_cramers_v.items(), key=lambda kv: -kv[1]):
    print(f"  Cramer's V({col}, binary KSI) = {v:.4f}")
```

Uses whichever already-loaded full-population frame and binary label series
the notebook has in scope at that point (the same one used to build
`y_train_bin`/`y_val_bin`/`y_test_bin` upstream — concatenated or the
pre-split source frame, whichever avoids reloading data). Exact variable
names are resolved during implementation by reading the notebook's existing
state at the insertion point; this is a read-only diagnostic, no new data
load.

**Cell B — Feature importances of the actual binary champion.**

```python
rf_step = pipeline_binary_final.named_steps.get("classify", pipeline_binary_final)
importances = getattr(rf_step, "feature_importances_", None)
if importances is not None:
    feature_names = pipeline_binary_final.named_steps["preprocess"].get_feature_names_out()
    top_importances = (
        pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
    )
    print(top_importances.to_string())
```

Guarded with `getattr(..., None)` because if Part 1 promotes a calibrated
pipeline (`CalibratedClassifierCV` wraps the estimator and does not expose
`feature_importances_` directly), this cell degrades gracefully to "not
available for the calibrated wrapper" rather than raising.

**Cell C — Markdown: evidence and literature context.** States, in prose:
the fresh Cramér's V numbers from Cell A, the top features from Cell B, the
three literature comparison points (Santos 2022, Pakgohar 2021, Schlößler
2024) with the existing caveat about covariate/leakage comparability from
`Technical_Review_Next_Steps.md` §6/4.1, and one paragraph stating plainly
that Test-2024 = 0.6026 is consistent with this evidence and with published
work on the same problem framing — not a shortfall.

## Data flow

No new data sources. Part 1 reuses `X_sub`/`y_sub_bin`/`groups_sub` (or the
SVM-specific subsample if `binary_champion_family` is `svm_rbf`) and
`BINARY_STAGE1_DATA[binary_champion_family]` exactly as `## 16`/`## 17`
already do. Part 2 reuses the already-loaded feature frame and the binary
label already constructed upstream for `y_train_bin`/`y_val_bin`/
`y_test_bin`.

## Testing

- No new library functions are added (only new call sites of
  `select_best_candidate`, `find_best_binary_threshold`,
  `CalibratedClassifierCV`, all of which are either already
  covered by existing tests or are third-party/sklearn code) — so no new
  unit test files are required by this plan.
- Verification is end-to-end notebook execution (`jupytext --sync`, execute
  all cells, assert zero errored cells) plus `pytest -q`, `ruff check`,
  `ruff format --check` on any touched `.py`/`.ipynb` mirror, matching the
  verification discipline used for the prior binary-champion-search plan.
- The promotion-gate logic is simple enough (two `>=` comparisons) that a
  live execution showing the printed promote/no-promote branch and its
  reasoning is sufficient evidence; no separate unit test is warranted for
  a one-time notebook branch.

## Risks

- **Wall-clock cost:** the multi-objective Optuna run repeats roughly the
  same CV budget as the existing single-objective tune (20 trials × 3-fold
  CV on the same subsample) — expect similar runtime to the existing `## 16`
  cell (order of minutes, per the prior run's logged timings). Calibration
  adds one additional full-training-set fit.
- **No regression risk to the current champion:** the promotion gate is
  strictly `>=` on both Val metrics against the existing champion's own Val
  numbers: it can only replace the champion with something at least as good
  on both axes, never worse.
- **Test-set discipline:** only the final selected pipeline (whichever one)
  is evaluated on Test-2024, exactly once — preserving the "one touch"
  property the prior technical review specifically flagged as a strength of
  this project's methodology.
