# Binary KSI Front Refinement & Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attempt one bounded, evidence-backed refinement of the binary KSI champion (multi-objective tuning + calibration, promoted only if it strictly improves on the current Val-2023 operating point) and add a binary-target-specific evidence section (fresh Cramér's V, feature importances, literature context) so the notebook's binary result is as rigorously justified as the existing 3-class ceiling argument.

**Architecture:** All work is inside `notebooks/03_A3_Phase.ipynb` (mirrored to `notebooks/03_A3_Phase.py` via jupytext). Two new sections are inserted between the existing `## 17` (refit/threshold/Test-2024) and the results summary, reusing existing library functions (`select_best_candidate`, `find_best_binary_threshold`, `recall_for_class`, `evaluate_binary_predictions`, `meets_binary_acceptance_criteria`) rather than adding new ones. **Every insertion task renumbers its downstream sections in the same script, so the notebook has unique, sequential section numbers after every single commit** — never a two-step "insert now, renumber later" split, which was tried and found to leave a duplicate section number between commits (see Task 1/2 design note below).

**Tech Stack:** Python, scikit-learn (`CalibratedClassifierCV`), Optuna (multi-objective `TPESampler`), scipy (`chi2_contingency`), pandas, jupytext, nbformat.

## Global Constraints

- **Never hand-edit `notebooks/03_A3_Phase.py` directly.** It is a jupytext mirror of `notebooks/03_A3_Phase.ipynb`. Every notebook change is made by writing a small throwaway Python script that loads the `.ipynb` with `nbformat`, mutates `cells[i]["source"]`, and saves it back — then running `uv run jupytext --sync notebooks/03_A3_Phase.ipynb` to regenerate the `.py` mirror. Validate the result with `python3 -c "import ast; ast.parse(open('notebooks/03_A3_Phase.py').read())"` after every sync.
- **`.ipynb` markdown-cell `source` strings have no `# ` prefix** — that prefix only appears in the `.py` mirror representation. Write raw markdown text (e.g. `"## 18 — ..."`, not `"# ## 18 — ..."`) into `.ipynb` cell sources.
- **Section numbers must stay unique and sequential after every commit, not just at the end of the plan.** The notebook already has a legitimate `## 0 — Setup and reproducibility` section, so the full sequence starts at 0, not 1 — verification scripts in this plan check `nums == list(range(nums[0], nums[-1] + 1))`, not `range(1, ...)`. Both notebook-editing tasks (1 and 2) insert a new section AND renumber the sections after it in the same script/commit — this was validated by dry-running the exact scripts against a scratch copy of the notebook before finalizing this plan; a two-step "insert then renumber later" split was tried first and produces a duplicate section number between the two commits, which is the same "taped together" defect flagged in the prior SVM plan review.
- **Test-2024 is touched at most once per pipeline that is actually selected as final.** The multi-objective candidate is only evaluated on Test-2024 if it is promoted (Task 1's promotion gate). A non-promoted candidate's Test-2024 metrics are never computed.
- **The promotion gate compares Val-2023 metrics to Val-2023 metrics only** — never CV-fold metrics to Val metrics, and never Test metrics as part of the promotion decision.
- **No new files, no new shared library functions.** Every new call in this plan is either a new notebook cell using existing `src/unfallatlas/` functions and third-party libraries (`CalibratedClassifierCV`, `optuna`, `chi2_contingency`), or a data-only field added to the existing binary model card JSON.
- **Do not touch** the 3-class notebook path, 3-class artifacts (`a3_best_model.joblib`, `a3_model_card.json`, `a3_model_comparison.csv`), or any file outside `notebooks/03_A3_Phase.ipynb`/`.py` and the three binary artifact files (`a3_binary_best_model.joblib`, `a3_binary_model_card.json`, `a3_binary_model_comparison.csv`).
- Verification after every task: `uv run jupytext --sync notebooks/03_A3_Phase.ipynb` (no diff beyond the intended cells), `ast.parse` on the `.py` mirror, `uv run ruff check notebooks/03_A3_Phase.py`, `uv run ruff format --check notebooks/03_A3_Phase.py`. Full end-to-end notebook execution happens once, in Task 3.

---

### Task 1: Insert `## 18 — Binary Multi-Objective Tuning & Calibration Refinement`, renumber downstream sections, wire model card field

**Files:**
- Modify: `notebooks/03_A3_Phase.ipynb` (via nbformat script) — in one script: (a) retitle the markdown cell currently reading `"## 18 — Binary Artifacts: Save Pipeline & Model Card"` to `"## 19 — Binary Artifacts: Save Pipeline & Model Card"`; (b) retitle the markdown cell currently reading `"## 19 — Results Summary: Binary KSI Classification"` to `"## 20 — Results Summary: Binary KSI Classification"`; (c) add a `"multiobjective_refinement": multiobj_refinement_record,` field to the `binary_model_card` dict literal in the artifacts-save code cell; (d) insert one new markdown cell + one new code cell, titled `## 18 — Binary Multi-Objective Tuning & Calibration Refinement`, immediately before the now-renumbered `## 19` artifacts cell.
- Modify: `notebooks/03_A3_Phase.py` — regenerated by `jupytext --sync`, not hand-edited.

**Interfaces:**
- Consumes: `binary_champion_family`, `BINARY_PARAM_SPACES` (via the already-bound `param_space_fn`), `BINARY_CV_DATA` (via `X_cv`/`y_cv`/`groups_cv`), `build_champion_fn`, `_binary_fit_kwargs`, `BINARY_STAGE1_DATA`, `X_val_bin`, `y_val_bin`, `X_test_bin`, `y_test_bin`, `best_threshold`, `best_f1_val`, `best_val_metrics`, `pipeline_binary_final`, `metrics_binary_test`, `gate_passed`, `best_params`, `OPTUNA_TRIALS`, `SEED` (all already defined by the existing `## 15`/`## 16`/`## 17` sections — read them, do not redefine). `recall_for_class`, `select_best_candidate`, `evaluate_binary_predictions`, `meets_binary_acceptance_criteria`, `find_best_binary_threshold`, `f1_score`, `optuna`, `GroupKFold`, `np`, `pd` (all already imported earlier in the notebook).
- Produces: `multiobj_refinement_record: dict` (consumed by the same task's model-card-field edit). If promoted, this task's inserted cell reassigns `pipeline_binary_final`, `best_threshold`, `best_params`, `best_f1_val`, `best_val_metrics`, `metrics_binary_test`, `gate_passed` in place — the renumbered `## 19` save cell and the eventual `## 21` results-summary numbers (Task 2) must reflect whichever values are current after this cell runs, with no separate "which pipeline do I save" branch needed downstream.

- [ ] **Step 1: Write the combined nbformat script**

Create a throwaway script (not committed) at `/tmp/task1_combined.py`:

```python
import nbformat

path = "notebooks/03_A3_Phase.ipynb"
nb = nbformat.read(path, as_version=4)

# --- Step A: renumber the two downstream sections FIRST (so the insertion
# point lookup below still matches "## 18 — Binary Artifacts" unambiguously) ---
renumbered_artifacts = False
renumbered_results = False
for cell in nb.cells:
    if cell.cell_type != "markdown":
        continue
    stripped = cell.source.strip()
    if stripped.startswith("## 18 — Binary Artifacts: Save Pipeline & Model Card"):
        cell.source = cell.source.replace(
            "## 18 — Binary Artifacts: Save Pipeline & Model Card",
            "## 19 — Binary Artifacts: Save Pipeline & Model Card",
        )
        renumbered_artifacts = True
    elif stripped.startswith("## 19 — Results Summary: Binary KSI Classification"):
        cell.source = cell.source.replace(
            "## 19 — Results Summary: Binary KSI Classification",
            "## 20 — Results Summary: Binary KSI Classification",
        )
        renumbered_results = True
assert renumbered_artifacts, "Could not find '## 18 — Binary Artifacts' to renumber"
assert renumbered_results, "Could not find '## 19 — Results Summary' to renumber"

# --- Step B: add the multiobjective_refinement field to the model card cell ---
card_field_added = False
for cell in nb.cells:
    if cell.cell_type == "code" and '"model_type": "binary_ksi_vs_slight"' in cell.source:
        marker = '"acceptance_gate_passed": bool(gate_passed),'
        assert marker in cell.source, "Expected marker line not found in model-card cell"
        cell.source = cell.source.replace(
            marker,
            marker + '\n    "multiobjective_refinement": multiobj_refinement_record,',
        )
        card_field_added = True
        break
assert card_field_added, "Could not find the binary_model_card cell to patch"

# --- Step C: insert the new ## 18 section immediately before the (now ## 19) artifacts section ---
insert_idx = None
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "markdown" and cell.source.strip().startswith(
        "## 19 — Binary Artifacts: Save Pipeline & Model Card"
    ):
        insert_idx = i
        break
assert insert_idx is not None, "Could not find the renumbered '## 19 — Binary Artifacts' cell"

markdown_source = """## 18 — Binary Multi-Objective Tuning & Calibration Refinement

The technical review (`docs/project/Technical_Review_Next_Steps.md`, categories 1.4 and 2.3)
identified two levers never tried for the binary champion: tuning that optimises macro-F1 and
Recall(KSI) jointly (Pareto-aware) instead of macro-F1 alone, and probability calibration before
threshold search. This section attempts both, on `binary_champion_family` only (SS15's winner),
reusing the same param space, CV data, and 20-trial budget as SS16's single-objective tune.

The multi-objective candidate is promoted to champion **only if it is at least as good as the
current champion on both Val-2023 macro-F1 and Val-2023 Recall(KSI)** — a strict non-regression
gate, never a trade-off. If it does not clear the gate, the negative result is reported honestly
and the SS16/SS17 champion is kept unchanged. Test-2024 is evaluated at most once, for whichever
pipeline ends up being final."""

markdown_cell = nbformat.v4.new_markdown_cell(source=markdown_source)

code_source = '''from sklearn.calibration import CalibratedClassifierCV  # noqa: E402

from unfallatlas.models.evaluate import BINARY_RECALL_KSI_THRESHOLD  # noqa: E402


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
        fold_recall.append(recall_for_class(y_cv.iloc[va_idx], pred, target_class=1))
    return float(np.mean(fold_f1)), float(np.mean(fold_recall))


study_binary_mo = optuna.create_study(
    directions=["maximize", "maximize"],
    sampler=optuna.samplers.TPESampler(seed=SEED),
    study_name=f"binary_{binary_champion_family}_multiobj",
)
study_binary_mo.optimize(binary_multiobj_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

print(f"Multi-objective Pareto front: {len(study_binary_mo.best_trials)} trial(s).")

pareto_rows = pd.DataFrame(
    [
        {"params": t.params, "macro_f1": t.values[0], "recall_ksi": t.values[1]}
        for t in study_binary_mo.best_trials
    ]
)
pareto_best_row = select_best_candidate(
    pareto_rows, recall_threshold=BINARY_RECALL_KSI_THRESHOLD, recall_col="recall_ksi"
)
pareto_best_params = pareto_best_row["params"]
print(f"Pareto-selected params: {pareto_best_params}")
print(
    f"Pareto-selected CV macro_f1={pareto_best_row['macro_f1']:.4f}  "
    f"CV recall_ksi={pareto_best_row['recall_ksi']:.4f}"
)

X_refit_mo, y_refit_mo = BINARY_STAGE1_DATA[binary_champion_family]
pipeline_binary_multiobj = build_champion_fn()
pipeline_binary_multiobj.set_params(
    **{f"classify__{k}": v for k, v in pareto_best_params.items()}
)
refit_fit_kwargs_mo = _binary_fit_kwargs(binary_champion_family, y_refit_mo)

pipeline_binary_calibrated = CalibratedClassifierCV(
    pipeline_binary_multiobj, method="isotonic", cv=3
)
pipeline_binary_calibrated.fit(X_refit_mo, y_refit_mo, **refit_fit_kwargs_mo)
print(f"Calibrated refit of {binary_champion_family} on {len(y_refit_mo):,} rows complete.")

y_val_scores_mo = pipeline_binary_calibrated.predict_proba(X_val_bin)[:, 1]
best_threshold_mo, best_val_metrics_mo = find_best_binary_threshold(y_val_bin.values, y_val_scores_mo)

print(f"\\nMulti-objective candidate — Val-2023 threshold: {best_threshold_mo:.4f}")
print(f"Val-2023 macro-F1: {best_val_metrics_mo['macro_f1']:.4f}")
print(f"Val-2023 recall(KSI): {best_val_metrics_mo['recall_ksi']:.4f}")

promote_multiobj = (
    best_val_metrics_mo["macro_f1"] >= best_f1_val
    and best_val_metrics_mo["recall_ksi"] >= best_val_metrics["recall_ksi"]
)

if promote_multiobj:
    print(
        f"\\nMulti-objective tuning + calibration improved the Val-2023 operating point "
        f"(macro_f1 {best_f1_val:.4f} -> {best_val_metrics_mo['macro_f1']:.4f}, "
        f"recall_ksi {best_val_metrics['recall_ksi']:.4f} -> "
        f"{best_val_metrics_mo['recall_ksi']:.4f}); promoted to champion."
    )
    pipeline_binary_final = pipeline_binary_calibrated
    best_threshold = best_threshold_mo
    best_params = pareto_best_params
    best_f1_val = best_val_metrics_mo["macro_f1"]
    best_val_metrics = best_val_metrics_mo

    y_test_scores_bin = pipeline_binary_final.predict_proba(X_test_bin)[:, 1]
    y_test_pred_bin = (y_test_scores_bin >= best_threshold).astype(int)
    metrics_binary_test = evaluate_binary_predictions(y_test_bin.values, y_test_pred_bin)
    gate_passed = meets_binary_acceptance_criteria(metrics_binary_test)

    print("\\nBinary KSI — Test-2024 metrics (multi-objective champion):")
    for k, v in metrics_binary_test.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}")
    print(f"Binary gate passed: {gate_passed}")
else:
    print(
        f"\\nMulti-objective tuning + calibration did not improve on the Val-2023 front "
        f"(macro_f1={best_val_metrics_mo['macro_f1']:.4f} vs {best_f1_val:.4f}, "
        f"recall_ksi={best_val_metrics_mo['recall_ksi']:.4f} vs "
        f"{best_val_metrics['recall_ksi']:.4f}); keeping the single-objective champion."
    )

multiobj_refinement_record = {
    "attempted": True,
    "promoted": bool(promote_multiobj),
    "val_macro_f1": float(best_val_metrics_mo["macro_f1"]),
    "val_recall_ksi": float(best_val_metrics_mo["recall_ksi"]),
    "promotion_rule": (
        "promoted iff multiobj Val-2023 macro_f1 >= single-objective champion's Val-2023 "
        "macro_f1 AND multiobj Val-2023 recall_ksi >= single-objective champion's Val-2023 "
        "recall_ksi (strict non-regression on both axes, Val-to-Val comparison only)"
    ),
}'''

code_cell = nbformat.v4.new_code_cell(source=code_source)

nb.cells.insert(insert_idx, code_cell)
nb.cells.insert(insert_idx, markdown_cell)

nbformat.write(nb, path)
print(f"Renumbered ## 18->19, ## 19->20; inserted new ## 18 before index {insert_idx}. Cell count: {len(nb.cells)}")
```

This exact script was dry-run against a scratch copy of the current notebook during planning and confirmed to: find both target cells, add the model-card field with valid resulting Python syntax, and insert the new section — producing a fully sequential `## 0 .. ## 20` numbering with no duplicates.

- [ ] **Step 2: Run the script and sync**

```bash
uv run python /tmp/task1_combined.py
uv run jupytext --sync notebooks/03_A3_Phase.ipynb
python3 -c "import ast; ast.parse(open('notebooks/03_A3_Phase.py').read())"
```

Expected: script prints `Renumbered ## 18->19, ## 19->20; inserted new ## 18 before index <N>. Cell count: <old+2>`; `ast.parse` raises no exception.

- [ ] **Step 3: Verify section numbers are unique and sequential**

```bash
python3 -c "
import re
content = open('notebooks/03_A3_Phase.py').read()
nums = [int(m.group(1)) for m in re.finditer(r'^# ## (\d+) ', content, re.MULTILINE)]
assert nums == sorted(set(nums)), f'duplicate or out-of-order section numbers: {nums}'
assert nums == list(range(nums[0], nums[-1] + 1)), f'gap in section numbers: {nums}'
print('OK', nums)
"
```

Expected: `OK [0, 1, 2, ..., 20]` (the notebook's real `## 0 — Setup and reproducibility` section means the sequence starts at 0, not 1) — no gaps or duplicates.

- [ ] **Step 4: Verify the model card and multiobj cells are present and syntactically valid**

```bash
python3 -c "
content = open('notebooks/03_A3_Phase.py').read()
assert '## 18 — Binary Multi-Objective Tuning & Calibration Refinement' in content
assert '## 19 — Binary Artifacts' in content
assert '## 20 — Results Summary' in content
assert '\"multiobjective_refinement\": multiobj_refinement_record,' in content
print('OK')
"
```

- [ ] **Step 5: Lint check**

```bash
uv run ruff check notebooks/03_A3_Phase.py
uv run ruff format --check notebooks/03_A3_Phase.py
```

Expected: both clean. If `ruff format --check` fails only on the newly inserted lines, run `uv run ruff format notebooks/03_A3_Phase.py` then re-run `uv run jupytext --sync notebooks/03_A3_Phase.ipynb` to write the formatted version back into the `.ipynb` (jupytext sync is bidirectional — syncing after formatting the `.py` side propagates the formatting into the `.ipynb` cell source).

- [ ] **Step 6: Commit**

```bash
git add notebooks/03_A3_Phase.ipynb notebooks/03_A3_Phase.py
git commit -m "feat(notebook/a3): add multi-objective tuning + calibration refinement for binary champion"
```

---

### Task 2: Insert `## 20 — Binary KSI Evidence: Association & Feature Importance` and renumber Results Summary

**Files:**
- Modify: `notebooks/03_A3_Phase.ipynb` (via nbformat script) — in one script: (a) retitle the markdown cell currently reading `"## 20 — Results Summary: Binary KSI Classification"` to `"## 21 — Results Summary: Binary KSI Classification"` and append two `<!-- FILL IN FROM LIVE RUN -->` bullet markers to it; (b) insert one new markdown cell + two new code cells + one new markdown cell, titled `## 20 — Binary KSI Evidence: Association & Feature Importance`, immediately before the now-renumbered `## 21` results-summary cell.
- Modify: `notebooks/03_A3_Phase.py` — regenerated by `jupytext --sync`.

**Interfaces:**
- Consumes: `df` (the full training frame loaded by `load_training_frame(BASE)` earlier in the notebook — already in scope), `pipeline_binary_final` (from Task 1, possibly the promoted calibrated pipeline), `np`, `pd`.
- Produces: `binary_cramers_v: dict[str, float]`, `binary_top_importances: pd.Series | None` — both are printed for the record; consumed only by hand-written prose in Task 3's fill-in step, not by any later code.

- [ ] **Step 1: Write the combined nbformat script**

Create a throwaway script (not committed) at `/tmp/task2_combined.py`:

```python
import nbformat

path = "notebooks/03_A3_Phase.ipynb"
nb = nbformat.read(path, as_version=4)

# --- Step A: renumber Results Summary FIRST (20 -> 21) and add fill-in markers ---
renumbered_results = False
for cell in nb.cells:
    if cell.cell_type == "markdown" and cell.source.strip().startswith(
        "## 20 — Results Summary: Binary KSI Classification"
    ):
        cell.source = cell.source.replace(
            "## 20 — Results Summary: Binary KSI Classification",
            "## 21 — Results Summary: Binary KSI Classification",
        )
        addition = """

- **Multi-objective tuning + calibration refinement (SS18)**: <!-- FILL IN FROM LIVE RUN: state
  whether the multi-objective/calibration candidate was promoted, and give its Val-2023
  macro-F1/recall_ksi vs. the single-objective champion's -->
- **Binary-target evidence (SS20)**: <!-- FILL IN FROM LIVE RUN: state the strongest binary-label
  Cramer's V value and name the feature, and note whether the Test-2024 result is consistent with
  the cited literature range -->"""
        cell.source = cell.source.rstrip() + addition
        renumbered_results = True
        break
assert renumbered_results, "Could not find '## 20 — Results Summary' to renumber"

# --- Step B: insert new ## 20 evidence section before the (now ## 21) results summary ---
insert_idx = None
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "markdown" and cell.source.strip().startswith(
        "## 21 — Results Summary: Binary KSI Classification"
    ):
        insert_idx = i
        break
assert insert_idx is not None, "Could not find the renumbered '## 21 — Results Summary' cell"

markdown_source = """## 20 — Binary KSI Evidence: Association & Feature Importance

`docs/project/Technical_Review_Next_Steps.md` established Cramer's V <= 0.13 for the strongest
available feature against the *3-class* `UKATGEORIE` target. This section repeats that association
check directly against the binary KSI label used by this model, and reports the actual binary
champion's feature importances (when available), so the Test-2024 result below is contextualised
against the same feature-limitation evidence used for the 3-class ceiling argument (SS11), rather
than an inference carried over from the 3-class analysis."""

markdown_cell = nbformat.v4.new_markdown_cell(source=markdown_source)

cell_a_source = '''from scipy.stats import chi2_contingency  # noqa: E402

from unfallatlas.features.preprocessing import ONEHOT_COLUMNS  # noqa: E402


def cramers_v(feature: pd.Series, target: pd.Series) -> float:
    """Bias-corrected Cramer's V (Bergsma & Wicher, 2013) - same formula SS6 of the
    U-phase notebook uses for the 3-class target, applied here to the binary label."""
    confusion = pd.crosstab(feature, target)
    n = confusion.values.sum()
    if n == 0:
        return float("nan")
    chi2 = chi2_contingency(confusion, correction=False)[0]
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2c = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rc = r - ((r - 1) ** 2) / (n - 1)
    kc = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float("nan") if denom <= 0 else float(np.sqrt(phi2c / denom))


y_binary_ksi_full = (df["UKATGEORIE"].astype(int) <= 2).astype(int)
binary_cramers_v = {
    col: cramers_v(df[col], y_binary_ksi_full) for col in ONEHOT_COLUMNS if col in df.columns
}
print("Cramer's V against the binary KSI label:")
for col, v in sorted(binary_cramers_v.items(), key=lambda kv: -kv[1]):
    print(f"  {col}: {v:.4f}")'''

cell_a = nbformat.v4.new_code_cell(source=cell_a_source)

cell_b_source = '''classify_step = None
if hasattr(pipeline_binary_final, "named_steps"):
    classify_step = pipeline_binary_final.named_steps.get("classify")

binary_top_importances = None
importances = getattr(classify_step, "feature_importances_", None) if classify_step is not None else None
if importances is not None:
    feature_names = pipeline_binary_final.named_steps["preprocess"].get_feature_names_out()
    binary_top_importances = (
        pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
    )
    print("Top 15 feature importances (binary champion):")
    print(binary_top_importances.to_string())
else:
    print(
        "Feature importances not available for this pipeline "
        "(calibrated wrapper or non-tree champion family)."
    )'''

cell_b = nbformat.v4.new_code_cell(source=cell_b_source)

cell_c_source = """### Evidence summary

- **Association with the binary label.** The Cramer's V values above repeat the U-phase's
  association check directly against KSI-vs-slight rather than the 3-class label. If the strongest
  value is still well below the ~0.3-0.5 range typically needed for strong classification signal,
  this confirms — on this exact target — the same feature-limitation the 3-class ceiling argument
  (SS11) relies on: the available Unfallatlas columns (accident type/context, weather, road surface)
  carry only weak marginal association with injury severity, because the actual physical determinants
  (impact speed, occupant age, seatbelt use, vehicle mass) are not present in the public dataset.
- **Feature importances** (when available) show which of the weakly-associated features the
  champion leans on most; they cannot exceed the ceiling implied by the association numbers above no
  matter how the model weights them.
- **Literature context.** Comparable KSI-vs-slight studies report macro-F1 in the 0.60-0.65 range
  (Santos, 2022, ~0.60; Pakgohar, 2021, ~0.62; Schlossler, 2024, ~0.65), though `Technical_Review_
  Next_Steps.md` (S6, item 6) notes these are not perfectly comparable — they often include
  person/vehicle-level covariates unavailable here, or use different resampling/leakage conventions.
  This project's Test-2024 macro-F1 (see SS21) falls inside that published range, which is consistent
  with — not short of — the state of the art for this problem framing on comparable feature sets."""

cell_c = nbformat.v4.new_markdown_cell(source=cell_c_source)

for cell in (cell_c, cell_b, cell_a, markdown_cell):
    nb.cells.insert(insert_idx, cell)

nbformat.write(nb, path)
print(f"Renumbered ## 20->21; inserted new ## 20 before index {insert_idx}. Cell count: {len(nb.cells)}")
```

This exact script was dry-run against a scratch copy during planning (chained after Task 1's script) and confirmed to produce fully sequential `## 0 .. ## 21` numbering with no duplicates, and a syntactically valid `.py` mirror.

- [ ] **Step 2: Run the script and sync**

```bash
uv run python /tmp/task2_combined.py
uv run jupytext --sync notebooks/03_A3_Phase.ipynb
python3 -c "import ast; ast.parse(open('notebooks/03_A3_Phase.py').read())"
```

- [ ] **Step 3: Verify section numbers are unique and sequential**

```bash
python3 -c "
import re
content = open('notebooks/03_A3_Phase.py').read()
nums = [int(m.group(1)) for m in re.finditer(r'^# ## (\d+) ', content, re.MULTILINE)]
assert nums == sorted(set(nums)), f'duplicate or out-of-order section numbers: {nums}'
assert nums == list(range(nums[0], nums[-1] + 1)), f'gap in section numbers: {nums}'
print('OK', nums)
"
```

Expected: `OK [0, 1, 2, ..., 21]`.

- [ ] **Step 4: Lint check**

```bash
uv run ruff check notebooks/03_A3_Phase.py
uv run ruff format --check notebooks/03_A3_Phase.py
```

- [ ] **Step 5: Commit**

```bash
git add notebooks/03_A3_Phase.ipynb notebooks/03_A3_Phase.py
git commit -m "feat(notebook/a3): add binary-target Cramer's V and feature-importance evidence section"
```

---

### Task 3: Execute end-to-end, fill in results, verify, commit

**Files:**
- Modify: `notebooks/03_A3_Phase.ipynb` / `.py` — fill in the two `<!-- FILL IN FROM LIVE RUN -->` markers from Task 2 with real numbers.
- Modify (only if Task 1's promotion gate fires during the live run): `data/processed/a3_binary_best_model.joblib`, `data/processed/a3_binary_model_card.json`, `data/processed/a3_binary_model_comparison.csv`.

**Interfaces:**
- Consumes: the fully assembled notebook from Tasks 1–2.
- Produces: final committed state.

- [ ] **Step 1: Execute the notebook end-to-end**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_A3_Phase.ipynb
```

If this is impractical inside the current environment (long-running), use the project's established live-kernel execution approach from the prior SVM plan (execute via `nbformat`/`jupyter_client` cell-by-cell) — whichever mechanism was used to produce the currently-committed real results is acceptable here; the goal is a genuine execution, not a re-use of stale outputs.

- [ ] **Step 2: Check for errored cells**

```bash
python3 -c "
import nbformat
nb = nbformat.read('notebooks/03_A3_Phase.ipynb', as_version=4)
errored = [c.execution_count for c in nb.cells if c.cell_type == 'code' for o in c.get('outputs', []) if o.get('output_type') == 'error']
print('errored cells:', errored)
print('total:', len(nb.cells))
assert not errored
"
```

Expected: `errored cells: []`.

- [ ] **Step 3: Read the printed output of the new §18 and §20 cells**

Capture the executed notebook's cell outputs (matching the prior plan's `/tmp/binary_outputs.txt`-style convention) to read: the promotion decision and both Val-2023 metric pairs from §18, and the Cramér's V values / top feature importances from §20.

- [ ] **Step 4: Fill in the two markers from Task 2 with the real numbers**

Write a throwaway nbformat script that replaces the two `<!-- FILL IN FROM LIVE RUN: ... -->` HTML comments in the `## 21` markdown cell with real prose built from Step 3's captured numbers — same technique used for every other `<!-- FILL IN -->` replacement in this notebook (locate the markdown cell by its section title, `str.replace` the exact comment text, write back, then `jupytext --sync`).

- [ ] **Step 5: If Task 1's gate promoted a new champion, verify the new artifacts**

```bash
python3 -c "
import joblib
p = joblib.load('data/processed/a3_binary_best_model.joblib')
print(type(p))
"
python3 -c "
import json
card = json.load(open('data/processed/a3_binary_model_card.json'))
print(card['multiobjective_refinement'])
print(card['acceptance_gate_passed'])
"
```

Expected: the model card's `multiobjective_refinement.promoted` matches what was printed during execution, and `acceptance_gate_passed` is `true`.

- [ ] **Step 6: Full verification suite**

```bash
uv run pytest -q
uv run ruff check notebooks/03_A3_Phase.py src/unfallatlas
uv run ruff format --check notebooks/03_A3_Phase.py src/unfallatlas
python3 -c "
content = open('notebooks/03_A3_Phase.py').read()
assert 'FILL IN FROM LIVE RUN' not in content, 'unresolved FILL-IN marker remains'
print('OK, no unresolved markers')
"
```

Expected: all tests pass, ruff clean, no remaining `FILL IN FROM LIVE RUN` markers.

- [ ] **Step 7: Commit**

```bash
git add notebooks/03_A3_Phase.ipynb notebooks/03_A3_Phase.py
git add data/processed/a3_binary_best_model.joblib data/processed/a3_binary_model_card.json data/processed/a3_binary_model_comparison.csv
git commit -m "feat(notebook/a3): execute binary front-refinement end-to-end, record real results"
```

If Task 1's gate did not promote a new champion, the `data/processed/a3_binary_*` files will show no diff (or only float-noise from re-fitting the unchanged pipeline during the live run) — `git status` before staging to confirm what actually changed, and only stage files that genuinely differ.

---

## Post-plan documentation update (not a task — do after Task 3 lands)

Per the project's established convention (every prior A³ plan updates these two files), after Task 3 is verified and committed:

- Add a new entry to `docs/prompts/03_prompts_phase_a3.md` linking this plan (`docs/superpowers/plans/2026-07-15-binary-ksi-front-refinement.md`) and summarizing the real outcome (promoted or not, and the evidence findings).
- Add one row to `docs/AI TOOL DISCLOSURE.md`'s "Detailed overview" table and one row to its "Implementation plan index" table, matching the existing row format for the `2026-07-14-svm-algorithm-selection.md` entry.
