# Presentation and Notebook Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one coherent English QUA³CK report whose charts are interactive Plotly figures,
whose notebook and exported HTML rendering agree, and whose C phase uses every persisted candidate
model where analytically valid.

**Architecture:** Move reusable visualization, artifact-registry, and sequential candidate-analysis
logic into focused library modules. Keep `.ipynb` files as the narrative source of truth, use saved
model checkpoints to avoid retraining, and validate both notebook structure and exported browser
behavior. Candidate selection remains validation-only; Test-2024 remains champion-only.

**Tech Stack:** Python 3.11+, Jupyter/nbformat/Jupytext, pandas, NumPy, scikit-learn, joblib,
Plotly 6.7, SHAP, nbconvert/Jinja2, BeautifulSoup, pytest, Ruff, Playwright, uv.

## Global Constraints

- The complete report, chart text, exporter chrome, and generated HTML are English.
- `notebooks/*.ipynb` are source of truth; never edit Jupytext `.py` mirrors directly.
- Preserve all existing uncommitted model, notebook-output, and generated-export work.
- Never stage or commit `notebooks/03_A3_Phase.ipynb`.
- Necessary console evidence remains visible; replace only raw dumps that a visual communicates
  more clearly.
- All charts use Plotly and export as saved `application/vnd.plotly.v1+json` output.
- Validation is the model comparison/selection surface; only the selected champion is analysed on
  Test-2024.
- Use current checkpoint artifacts; do not retrain merely to replace valid persisted models.
- Use `uv` for dependency and command execution.

---

### Task 1: Convert the shared F1/recall visualizations to Plotly

**Files:**
- Modify: `src/unfallatlas/viz/metrics_viz.py`
- Modify: `tests/test_metrics_viz.py`

**Interfaces:**
- Consumes: comparison frames with `model`, `macro_f1`, and either `recall_class_1` or
  `recall_ksi`.
- Produces:
  `plot_f1_recall_front(...) -> plotly.graph_objects.Figure` and
  `plot_binary_f1_recall_front(...) -> plotly.graph_objects.Figure`.

- [ ] **Step 1: Replace Matplotlib-oriented tests with failing Plotly contracts**

```python
import plotly.graph_objects as go


def test_plot_f1_recall_front_returns_interactive_plotly_figure(comparison_df):
    fig = plot_f1_recall_front(comparison_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert len(fig.data[0].x) == len(comparison_df)
    assert {shape.type for shape in fig.layout.shapes} == {"line", "rect"}


def test_plot_binary_front_uses_recall_ksi(binary_comparison_df):
    fig = plot_binary_f1_recall_front(binary_comparison_df)
    assert list(fig.data[0].x) == list(binary_comparison_df["recall_ksi"])
    assert list(fig.data[0].y) == list(binary_comparison_df["macro_f1"])
```

- [ ] **Step 2: Run the focused tests and confirm the old Axes API fails**

Run: `uv run pytest tests/test_metrics_viz.py -q`

Expected: failures because both helpers currently return `matplotlib.axes.Axes`.

- [ ] **Step 3: Implement one Plotly Pareto-front helper**

```python
def _plot_pareto_front(
    comparison_df: pd.DataFrame,
    *,
    recall_col: str,
    recall_axis_label: str,
    title: str,
    gate_f1: float,
    gate_recall: float,
    label_col: str,
) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=comparison_df[recall_col],
            y=comparison_df["macro_f1"],
            mode="markers+text",
            text=comparison_df[label_col],
            textposition="top center",
            customdata=comparison_df[[label_col]].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + recall_axis_label
                + ": %{x:.3f}<br>Macro-F1: %{y:.3f}<extra></extra>"
            ),
        )
    )
    fig.add_shape(
        type="rect",
        x0=gate_recall,
        x1=1,
        y0=gate_f1,
        y1=1,
        fillcolor="rgba(42, 157, 143, 0.12)",
        line_width=0,
        layer="below",
    )
    fig.add_vline(x=gate_recall, line_dash="dash", line_color="#E63946")
    fig.add_hline(y=gate_f1, line_dash="dash", line_color="#E63946")
    fig.update_layout(
        title=title,
        xaxis_title=recall_axis_label,
        yaxis_title="Macro-F1",
        template="plotly_white",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
    )
    return fig
```

Remove the Matplotlib import and the `ax` parameters. Preserve existing public names and gate
defaults.

- [ ] **Step 4: Run visualization tests**

Run: `uv run pytest tests/test_metrics_viz.py -q`

Expected: all tests pass and no Matplotlib backend setup remains.

- [ ] **Step 5: Commit the library conversion**

```bash
git add src/unfallatlas/viz/metrics_viz.py tests/test_metrics_viz.py
git commit -m "refactor(viz): standardize model fronts on plotly"
```

---

### Task 2: Add presentation rendering audits and English provenance chrome

**Files:**
- Modify: `src/unfallatlas/presentation/templates/notebook/index.html.j2`
- Modify: `src/unfallatlas/presentation/validation.py`
- Modify: `src/unfallatlas/presentation/rendering.py`
- Modify: `tests/presentation/test_rendering.py`
- Modify: `tests/presentation/test_validation.py`
- Modify: `tests/presentation/test_browser.py`

**Interfaces:**
- Produces:
  `validate_rendered_html(html: str) -> tuple[Finding, ...]`.
- `render_notebook` incorporates post-render findings and fails on malformed Markdown table
  paragraphs.

- [ ] **Step 1: Add failing tests for nested Markdown leakage and English chrome**

```python
def test_validate_rendered_html_detects_literal_markdown_table():
    html = "<main><p>| A | B |\\n|---|---|\\n| 1 | 2 |</p></main>"
    findings = validate_rendered_html(html)
    assert [f.code for f in findings] == ["literal-markdown-table"]


def test_template_moves_git_state_into_technical_provenance():
    soup = _render(_notebook())
    visible_header = soup.select_one(".document-header").get_text(" ", strip=True)
    assert "Arbeitsbaum" not in visible_header
    assert "Working tree" not in visible_header
    details = soup.select_one("details.technical-provenance")
    assert details
    assert "Uncommitted changes" in details.get_text(" ", strip=True)
```

Extend the Plotly browser test to assert that every `.plotly-graph-div` has a populated
`_fullLayout` after lazy loading and that browser console errors remain empty.

- [ ] **Step 2: Run the focused exporter tests**

Run:
`uv run pytest tests/presentation/test_rendering.py tests/presentation/test_validation.py -q`

Expected: the new tests fail because post-render validation and English technical provenance do
not exist.

- [ ] **Step 3: Implement rendered-HTML validation**

Use BeautifulSoup to inspect paragraph and list-item text. A finding is emitted only when a block
contains both a pipe-delimited row and a Markdown delimiter row matching
`^\s*\|?\s*:?-{3,}`. Do not flag ordinary prose containing a pipe.

```python
_MARKDOWN_TABLE_DELIMITER = re.compile(r"(?m)^\s*\|?\s*:?-{3,}.*\|\s*$")


def validate_rendered_html(html: str) -> tuple[Finding, ...]:
    soup = BeautifulSoup(html, "html.parser")
    findings = []
    for node in soup.select("p, li"):
        text = node.get_text("\n", strip=True)
        if "|" in text and _MARKDOWN_TABLE_DELIMITER.search(text):
            findings.append(
                _finding(
                    "error",
                    "literal-markdown-table",
                    "Markdown table syntax was emitted as paragraph text.",
                )
            )
    return tuple(findings)
```

Call this after the final exporter pass and before the atomic write. Treat an error finding as an
export error so malformed presentation HTML cannot be published silently.

- [ ] **Step 4: Simplify the visible header and add collapsed provenance**

Keep title, source path, export timestamp, readiness, warnings, execution state, cells, and outputs
in the audience-facing area. Move repository, commit, branch, and dirty state into:

```html
<details class="technical-provenance">
  <summary>Technical provenance</summary>
  <dl>
    <div><dt>Repository</dt><dd>...</dd></div>
    <div><dt>Commit</dt><dd>...</dd></div>
    <div><dt>Branch</dt><dd>...</dd></div>
    <div><dt>Uncommitted changes</dt><dd>Yes/No</dd></div>
  </dl>
</details>
```

- [ ] **Step 5: Run exporter and browser tests**

Run:
`uv run pytest tests/presentation/test_rendering.py tests/presentation/test_validation.py -q`

Then, if Playwright is installed:
`uv run pytest -m browser tests/presentation/test_browser.py -q`

Expected: structural tests pass; browser tests pass with locally loaded Plotly and no console
errors.

- [ ] **Step 6: Commit exporter changes**

```bash
git add src/unfallatlas/presentation tests/presentation
git commit -m "fix(presentation): validate rendering and clarify provenance"
```

---

### Task 3: Add a fingerprinted candidate-artifact registry

**Files:**
- Create: `src/unfallatlas/models/artifacts.py`
- Create: `tests/test_model_artifacts.py`
- Modify: `src/unfallatlas/models/__init__.py`

**Interfaces:**
- Produces:
  `CandidateArtifact` dataclass,
  `build_candidate_registry(checkpoint_dir, comparison_df) -> dict`,
  `validate_candidate_registry(registry, repo_root) -> list[CandidateArtifact]`, and
  `sha256_file(path) -> str`.

- [ ] **Step 1: Write failing registry tests**

```python
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
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest tests/test_model_artifacts.py -q`

Expected: import failure because `artifacts.py` does not exist.

- [ ] **Step 3: Implement the registry**

Use immutable `CandidateArtifact` fields:

```python
@dataclass(frozen=True)
class CandidateArtifact:
    model: str
    family: str
    path: Path
    sha256: str
    n_train: int | None
    score_interface: Literal["predict_proba", "decision_function", "predict"]
    evaluation_role: Literal["baseline", "candidate", "finalist", "champion"]
```

Map filenames directly from the comparison `model` field under `checkpoint_dir / "binary"`.
Classify Random Forest, XGBoost, LightGBM, and CatBoost as finalists; mark the model-card winner as
champion when the caller supplies `champion_model`. Store paths relative to the repository root.
Validation checks exact keys, containment inside the repository, file existence, and SHA-256.

- [ ] **Step 4: Run registry tests**

Run: `uv run pytest tests/test_model_artifacts.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit registry support**

```bash
git add src/unfallatlas/models/artifacts.py src/unfallatlas/models/__init__.py \
  tests/test_model_artifacts.py
git commit -m "feat(models): register persisted binary candidates"
```

---

### Task 4: Build sequential validation-only candidate analysis

**Files:**
- Create: `src/unfallatlas/models/candidate_analysis.py`
- Create: `tests/test_candidate_analysis.py`
- Modify: `src/unfallatlas/models/c_phase.py`
- Modify: `tests/test_c_phase.py`

**Interfaces:**
- Consumes: validated `CandidateArtifact` objects, `X_val`, `y_val`, and fixed probe samples.
- Produces:
  `candidate_scores(model, X) -> np.ndarray`,
  `measure_latency(model, X, repeats=5) -> float`,
  `measure_missing_feature_robustness(model, X, columns) -> dict`,
  `analyze_candidates(...) -> CandidateAnalysisResult`,
  `analysis_fingerprint(artifacts, data_fingerprint, parameters) -> str`,
  `load_or_analyze_candidates(...) -> CandidateAnalysisResult`,
  `prediction_disagreement(predictions) -> pd.DataFrame`, and
  `compute_finalist_permutation_importance(...) -> pd.DataFrame`.

- [ ] **Step 1: Add failing behavioral tests with small sklearn pipelines**

```python
def test_candidate_scores_supports_probability_and_margin_models():
    probability_model = LogisticRegression().fit(X, y)
    margin_model = LinearSVC().fit(X, y)
    assert candidate_scores(probability_model, X).shape == (len(X),)
    assert candidate_scores(margin_model, X).shape == (len(X),)


def test_analysis_never_receives_test_data(monkeypatch, artifacts, X_val, y_val):
    loaded = []

    def loader(path):
        loaded.append(path)
        return fitted_models[path.name]

    result = analyze_candidates(
        artifacts,
        X_val=X_val,
        y_val=y_val,
        latency_sample=X_val.iloc[:20],
        robustness_sample=X_val.iloc[:20],
        artifact_loader=loader,
    )
    assert len(result.metrics) == len(artifacts)
    assert set(result.scores) == {artifact.model for artifact in artifacts}
```

Also test that robustness reports prediction failure, mean absolute score drift, and changed-class
share rather than a fabricated scalar score.

- [ ] **Step 2: Add a failing stale-cache test**

```python
def test_load_or_analyze_recomputes_after_artifact_fingerprint_change(
    tmp_path, artifacts, X_val, y_val, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        candidate_analysis,
        "analyze_candidates",
        lambda *args, **kwargs: calls.append("computed") or expected_result,
    )
    load_or_analyze_candidates(
        artifacts,
        X_val=X_val,
        y_val=y_val,
        cache_dir=tmp_path,
        data_fingerprint="data-v1",
        parameters={"latency_repeats": 5, "random_state": 42},
    )
    changed = [replace(artifacts[0], sha256="different"), *artifacts[1:]]
    load_or_analyze_candidates(
        changed,
        X_val=X_val,
        y_val=y_val,
        cache_dir=tmp_path,
        data_fingerprint="data-v1",
        parameters={"latency_repeats": 5, "random_state": 42},
    )
    assert calls == ["computed", "computed"]
```

- [ ] **Step 3: Run focused tests**

Run:
`uv run pytest tests/test_candidate_analysis.py tests/test_c_phase.py -q`

Expected: failures because the analysis module and measured qualitative-input contract do not
exist.

- [ ] **Step 4: Implement score extraction and sequential analysis**

```python
def candidate_scores(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).reshape(-1)
    return np.asarray(model.predict(X), dtype=float).reshape(-1)
```

For every artifact: load, score Val-2023, evaluate default predictions, warm up, measure the median
of five latency repeats, run the missing-feature probe, retain compact arrays/results, then delete
the model and call `gc.collect()`.

- [ ] **Step 5: Implement the fingerprinted cache**

Canonicalize the ordered artifact names/SHA-256 values, data fingerprint, and analysis parameters
with `json.dumps(..., sort_keys=True, separators=(",", ":"))`, then hash the UTF-8 bytes.
`load_or_analyze_candidates` reuses the CSV/Parquet files only when
`c_phase_analysis_manifest.json` contains that exact fingerprint. A mismatch calls
`analyze_candidates` and atomically rewrites metrics, scores, and the manifest.

- [ ] **Step 6: Implement cross-model evidence**

`prediction_disagreement` returns pairwise disagreement shares at validation operating points.
Permutation importance uses `sklearn.inspection.permutation_importance` with
`scoring="f1_macro"`, `n_repeats=3`, `random_state=42`, and one fixed stratified validation sample.
Return tidy rows: `model`, `feature`, `importance_mean`, `importance_std`, `rank`.

- [ ] **Step 7: Replace the old qualitative scoring assumption**

Change `build_qualitative_matrix` to accept only columns explicitly present across compared rows.
Weights are normalized over available, varying criteria. Constant or missing criteria are reported
but do not affect the ranking.

- [ ] **Step 8: Run candidate-analysis tests**

Run:
`uv run pytest tests/test_candidate_analysis.py tests/test_c_phase.py -q`

Expected: all tests pass.

- [ ] **Step 9: Commit the analysis engine**

```bash
git add src/unfallatlas/models/candidate_analysis.py src/unfallatlas/models/c_phase.py \
  tests/test_candidate_analysis.py tests/test_c_phase.py
git commit -m "feat(c-phase): analyze persisted candidates on validation"
```

---

### Task 5: Add notebook narrative and visualization compliance tests

**Files:**
- Create: `tests/test_notebook_presentation_contract.py`

**Interfaces:**
- Consumes: the four `.ipynb` files directly through `nbformat`.
- Produces: executable assertions for language, ordering, checklists, Plotly usage, and saved MIME
  outputs.

- [ ] **Step 1: Write contract helpers and failing tests**

```python
NOTEBOOKS = [
    Path("notebooks/01_Q_Phase.ipynb"),
    Path("notebooks/02_U_Phase.ipynb"),
    Path("notebooks/03_A3_Phase.ipynb"),
    Path("notebooks/04_C_Phase.ipynb"),
]


def notebook_text(path: Path) -> str:
    notebook = nbformat.read(path, as_version=4)
    return "\n".join("".join(cell.source) for cell in notebook.cells)


def test_no_matplotlib_or_seaborn_chart_code():
    forbidden = ("matplotlib", "seaborn", "plt.", "sns.")
    for path in NOTEBOOKS:
        text = notebook_text(path)
        assert not any(token in text for token in forbidden), path


def test_no_development_checklists_or_stale_c_phase_limitation():
    combined = "\n".join(notebook_text(path) for path in NOTEBOOKS)
    assert "acceptance checklist" not in combined.lower()
    assert "[ ]" not in combined
    assert "only the champion pipeline" not in combined.lower()
    assert "pipelines were not persisted" not in combined.lower()


def test_a3_multiclass_decision_precedes_binary_search():
    text = notebook_text(Path("notebooks/03_A3_Phase.ipynb"))
    assert text.index("Three-class feasibility decision") < text.index(
        "Binary KSI candidate search"
    )
    assert text.count("A³ summary") == 1
```

Add an output test that treats an `image/png` output as a failure when the code cell contains a
known chart call. Do not reject non-chart evidence solely by MIME type.

- [ ] **Step 2: Run the tests and capture the current failures**

Run: `uv run pytest tests/test_notebook_presentation_contract.py -q`

Expected: failures for Matplotlib A³ cells, checklists, stale C prose, and mixed-language C text.

- [ ] **Step 3: Commit only the tests**

```bash
git add tests/test_notebook_presentation_contract.py
git commit -m "test(notebooks): define presentation quality contract"
```

---

### Task 6: Refactor the Q and U narratives

**Files:**
- Modify: `notebooks/01_Q_Phase.ipynb`
- Modify: `notebooks/02_U_Phase.ipynb`
- Regenerate: `notebooks/01_Q_Phase.py`
- Regenerate: `notebooks/02_U_Phase.py`

**Interfaces:**
- Produces: the target-policy narrative and target-independent preprocessing handoff consumed by
  A³.

- [ ] **Step 1: Reorganize Q**

Move the gate-revision content into target definition and success metrics. Renumber the final
sections consecutively. State explicitly that the three-class question was original and KSI is the
evidence-driven operational revision. End with one summary and one U transition.

Use this final top-level order:

```text
1 Problem context
2 Research question
3 Hypotheses
4 Prediction goal
5 Target definition and staged feasibility policy
6 Unit of analysis and prediction horizon
7 Stakeholders and decision context
8 Success metrics and gates
9 Constraints
10 Data sources
11 Feasibility and literature anchor
12 Known limitations
13 Summary and U-phase handoff
```

- [ ] **Step 2: Reorganize U around target viability**

Keep the original three-class EDA, but connect imbalance, weak Cramér's V, stable severity shares,
and missing physical determinants. Introduce KSI as a fallback before the preprocessing handoff.
Remove the U acceptance checklist, duplicated interpretations, and stale A³ section references.

Keep the existing audit order, but end with:

```text
9 Leakage and chronological-split audit
10 Target viability and preprocessing decisions
11 Risks, decisions, and A³ handoff
```

- [ ] **Step 3: Keep Plotly and console evidence intact**

Do not suppress existing intentional prints. Ensure every U chart cell still calls a Plotly figure
and uses English titles/labels. Set:

```python
pio.renderers.default = "plotly_mimetype"
```

so saved outputs use the exporter-supported MIME representation.

- [ ] **Step 4: Sync mirrors and run contract tests**

Run: `uv run jupytext --sync notebooks/01_Q_Phase.ipynb notebooks/02_U_Phase.ipynb`

Run:
`uv run pytest tests/test_notebook_presentation_contract.py tests/presentation/test_rendering.py -q`

Expected: Q/U-specific contract assertions pass; A³/C assertions still fail.

- [ ] **Step 5: Commit Q/U source and mirrors**

```bash
git add notebooks/01_Q_Phase.ipynb notebooks/01_Q_Phase.py \
  notebooks/02_U_Phase.ipynb notebooks/02_U_Phase.py
git commit -m "docs(notebooks): align q and u target narrative"
```

---

### Task 7: Refactor A³ into the multiclass-to-binary decision funnel

**Files:**
- Modify but never stage: `notebooks/03_A3_Phase.ipynb`
- Regenerate: `notebooks/03_A3_Phase.py`
- Modify: `data/processed/a3_binary_model_card.json`

**Interfaces:**
- Consumes: Plotly metric helpers and candidate registry.
- Produces: one coherent A³ narrative, interactive chart outputs, and
  `model_card["checkpoint_id"]` plus `model_card["candidate_artifacts"]`.

- [ ] **Step 1: Reorder Markdown sections without changing scientific results**

Create Part I (three-class feasibility) and Part II (binary KSI model). Move the ceiling analysis
before the binary target definition. Merge the multi-objective/calibration negative result into the
binary tuning decision. Remove the old section 10 checklist/summary and retain one final summary.

Use this top-level order:

```text
0 Setup and reproducibility
1 Data contract and chronological evaluation protocol
Part I — Three-class feasibility
2 Baselines and candidate families
3 Imbalance strategies and tuning
4 Three-class validation result
5 Empirical and arithmetic ceiling
6 Three-class feasibility decision
Part II — Binary KSI modelling
7 Binary target and acceptance criteria
8 Stage 0/Stage 1 candidate search
9 Gate-aware validation selection
10 Winning-family tuning and calibration refinement
11 Full refit, validation threshold, and one Test-2024 evaluation
12 Binary evidence and persisted artifacts
13 A³ summary and C-phase handoff
```

- [ ] **Step 2: Replace A³.11 and A³.19 static figures**

Use the Plotly-returning helpers:

```python
fig = plot_f1_recall_front(comparison_df)
fig.write_html(out_path.with_suffix(".html"), include_plotlyjs=False)
fig.show()
```

and:

```python
fig = plot_binary_f1_recall_front(binary_comparison_df)
fig.write_html(binary_out_path.with_suffix(".html"), include_plotlyjs=False)
fig.show()
```

Remove Matplotlib imports, `savefig`, and PNG-only figure references.

- [ ] **Step 3: Replace raw evidence dumps where a Plotly view is clearer**

Convert the binary confusion-matrix print/table to `plot_confusion_matrix_heatmap`. Convert Cramér's
V and top feature importances to sorted horizontal Plotly bars. Keep concise threshold, checkpoint,
and metric prints that serve as execution evidence.

- [ ] **Step 4: Record every persisted candidate**

Record the model run separately from the documentation commit. Set
`model_card["checkpoint_id"] = "b1ea31e"` for the already-complete current run, call
`build_candidate_registry` with `a3_checkpoints/<checkpoint_id>`, and add the returned JSON object
to the binary model card. Validate it before writing the model card. Do not derive this directory
from the new documentation `HEAD`, because that would miss the valid checkpoints and retrain every
model after an unrelated commit.

- [ ] **Step 5: Make Optuna execution idempotent**

For each persisted study:

```python
completed_trials = sum(
    trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
)
remaining_trials = max(0, TARGET_TRIALS - completed_trials)
if remaining_trials:
    study.optimize(objective, n_trials=remaining_trials)
```

This prevents re-execution from adding another full trial budget.

- [ ] **Step 6: Fix the final Markdown summary**

Remove the nested confusion-matrix Markdown table entirely. The summary references the interactive
heatmap and contains no indented pipe table.

- [ ] **Step 7: Sync the mirror and run tests**

Run: `uv run jupytext --sync notebooks/03_A3_Phase.ipynb`

Run:
`uv run pytest tests/test_notebook_presentation_contract.py tests/test_metrics_viz.py \
tests/test_model_artifacts.py -q`

Expected: A³ structure and Plotly-source tests pass.

- [ ] **Step 8: Commit only the model-card schema update**

```bash
git add data/processed/a3_binary_model_card.json
git commit -m "refactor(a3): integrate binary decision narrative"
```

Before committing, verify `git diff --cached --name-only` does not contain
`notebooks/03_A3_Phase.ipynb` or `notebooks/03_A3_Phase.py`. Both source notebook and generated
mirror remain uncommitted because project policy forbids committing a changed mirror without its
matching notebook.

---

### Task 8: Finalize C with measured multi-model evidence

**Files:**
- Modify: `notebooks/04_C_Phase.ipynb`
- Regenerate: `notebooks/04_C_Phase.py`
- Create during execution: `data/processed/c_phase_candidate_metrics.csv`
- Create during execution: `data/processed/c_phase_candidate_scores.parquet`
- Create during execution: `data/processed/c_phase_permutation_importance.csv`
- Create during execution: `data/processed/c_phase_analysis_manifest.json`
- Modify during execution: `data/processed/c_phase_inference_contract.json`

**Interfaces:**
- Consumes: A³ model card registry and candidate-analysis functions.
- Produces: validation comparisons, champion-only test analysis, refreshed conclusions, and final
  K-phase inference contract.

- [ ] **Step 1: Rewrite the C setup and model-comparison sections in English**

Load and validate the registry. Remove the claim that runner-up pipelines are unavailable. Build
one all-candidate Plotly table/front and threshold-free ROC/PR curves for substantive models using
Val-2023 scores.

Use this top-level order:

```text
0 Scope, artifact registry, and evaluation contract
1 All-candidate Val-2023 comparison
2 ROC/PR and operating-point trade-offs
3 Measured finalist comparison
4 Champion-only Test-2024 confirmation and gate
5 Champion error analysis
6 Cross-model feature evidence and champion SHAP
7 Literature context and limitations
8 Final model decision and K-phase contract
9 C-phase summary
```

- [ ] **Step 2: Add measured finalist comparison**

Sequentially analyse Random Forest, XGBoost, LightGBM, and CatBoost. Show:

- latency with medians and consistent units;
- robustness as failure status, score drift, and changed-class share;
- pairwise prediction disagreement heatmap;
- permutation-importance rank comparison.

Do not reuse champion measurements in other rows.

- [ ] **Step 3: Keep Test-2024 champion-only**

Reload `a3_binary_best_model.joblib`, apply the selected validation threshold, verify model-card
metrics within the documented cache tolerance, and show the Plotly confusion matrix and error
slices. Do not request Test-2024 predictions from other candidates.

- [ ] **Step 4: Retain and contextualize champion SHAP**

Keep the interactive beeswarm, mean-absolute SHAP bar, and four waterfall cases. Update English
interpretation to distinguish champion-specific SHAP from model-agnostic finalist permutation
importance.

- [ ] **Step 5: Rebuild the qualitative decision and conclusions**

Rank only varying, measured criteria. Explain the macro-F1/Recall trade-off and whether Random
Forest remains preferred after latency, robustness, disagreement, and importance evidence. Update
limitations, literature comparison, final decision, and the K-phase contract from computed values.

- [ ] **Step 6: Sync and test**

Run: `uv run jupytext --sync notebooks/04_C_Phase.ipynb`

Run:
`uv run pytest tests/test_notebook_presentation_contract.py tests/test_candidate_analysis.py \
tests/test_c_phase.py -q`

Expected: all notebook contract and C-analysis tests pass.

- [ ] **Step 7: Commit C source before execution outputs are refreshed**

```bash
git add notebooks/04_C_Phase.ipynb notebooks/04_C_Phase.py
git commit -m "feat(c-phase): compare persisted candidate models"
```

---

### Task 9: Execute notebooks and regenerate the offline presentation

**Files:**
- Modify with saved outputs: `notebooks/02_U_Phase.ipynb`
- Modify with saved outputs but never stage: `notebooks/03_A3_Phase.ipynb`
- Modify with saved outputs: `notebooks/04_C_Phase.ipynb`
- Regenerate: `reports/presentation/**`
- Regenerate: `reports/figures/u_phase/**`
- Regenerate: `reports/figures/c_phase/**`
- Regenerate: `data/processed/c_phase_*`

**Interfaces:**
- Produces: executed saved output consumed by `scripts/export_notebooks.py`.

- [ ] **Step 1: Snapshot the dirty-file list**

Run:
`git -c filter.lfs.process= -c filter.lfs.required=false status --short`

Expected: the pre-existing A³/model/export changes plus planned source changes; no unexplained file
loss.

- [ ] **Step 2: Execute U**

Run:
`uv run jupyter nbconvert --to notebook --execute --inplace \
notebooks/02_U_Phase.ipynb`

Expected: the cached enrichment path is used, all Plotly figures render, and no cell errors remain.

- [ ] **Step 3: Execute A³ without retraining valid checkpoints**

Run:
`uv run jupyter nbconvert --to notebook --execute --inplace \
notebooks/03_A3_Phase.ipynb`

Expected: load-or-fit messages show reuse from recorded checkpoint `b1ea31e`, completed Optuna
studies add no excess trials, all Plotly cells save interactive MIME output, and the final gate
remains passed.

- [ ] **Step 4: Execute C**

Run:
`uv run jupyter nbconvert --to notebook --execute --inplace \
notebooks/04_C_Phase.ipynb`

Expected: all candidate artifacts validate, candidate analysis is sequential, Test-2024 is used
only by the champion path, SHAP completes, and the inference contract is regenerated.

- [ ] **Step 5: Export all notebooks**

Run:
`uv run python scripts/export_notebooks.py --all`

Expected: four successful exports, no literal-Markdown-table error, no missing local assets, and an
updated `reports/presentation/manifest.json`.

- [ ] **Step 6: Verify the A³ notebook remains uncommitted**

Run:
`git -c filter.lfs.process= -c filter.lfs.required=false status --short \
notebooks/03_A3_Phase.ipynb`

Expected: ` M notebooks/03_A3_Phase.ipynb`.

---

### Task 10: Run complete verification and final editorial audit

**Files:**
- Modify only if verification finds defects: files already listed above.
- Regenerate after any fix: affected notebook HTML and manifest.

**Interfaces:**
- Produces: evidence that the complete notebook and HTML presentation satisfies the design.

- [ ] **Step 1: Run formatting and lint**

Run: `uv run ruff format --check .`

Expected: exit 0.

Run: `uv run ruff check .`

Expected: exit 0.

- [ ] **Step 2: Run the complete non-browser suite**

Run: `uv run pytest`

Expected: all non-browser tests pass with coverage output and no unexpected skips.

- [ ] **Step 3: Run browser verification**

Run: `uv run pytest -m browser`

Expected: all Playwright tests pass at supported viewports; Plotly loads only from local assets;
every figure initializes; tables scroll correctly; no browser console errors occur.

- [ ] **Step 4: Audit generated HTML mechanically**

Run:
`rg -n '<p>\s*\||Arbeitsbaum|matplotlib|image/png|pipelines were not persisted|only the champion' \
reports/presentation/notebooks`

Expected: no matches.

Run:
`rg -n 'plotly-graph-div' reports/presentation/notebooks/02_U_Phase.html \
reports/presentation/notebooks/03_A3_Phase.html reports/presentation/notebooks/04_C_Phase.html`

Expected: interactive Plotly containers in all analytical phase exports.

- [ ] **Step 5: Perform the editorial pass in reading order**

Review Q → U → A³ → C in the exported HTML. Verify section numbering, transitions, term
definitions, chart titles, metric consistency, model names, thresholds, cross-references, and
conclusions against the model cards and derived C evidence. Correct every discrepancy and rerun the
affected notebook/export/tests.

- [ ] **Step 6: Commit final committable artifacts without A³**

```bash
git add reports/presentation reports/figures \
  data/processed/a3_binary_model_card.json \
  data/processed/c_phase_candidate_metrics.csv \
  data/processed/c_phase_candidate_scores.parquet \
  data/processed/c_phase_permutation_importance.csv \
  data/processed/c_phase_analysis_manifest.json \
  data/processed/c_phase_inference_contract.json \
  notebooks/01_Q_Phase.ipynb notebooks/01_Q_Phase.py \
  notebooks/02_U_Phase.ipynb notebooks/02_U_Phase.py \
  notebooks/04_C_Phase.ipynb notebooks/04_C_Phase.py
git diff --cached --name-only
git commit -m "chore(presentation): publish coherent interactive report"
```

The staged-name check must not list `notebooks/03_A3_Phase.ipynb` or
`notebooks/03_A3_Phase.py`. After the commit, confirm the executed A³ notebook and its matching
mirror remain modified and available for future HTML exports.

- [ ] **Step 7: Record final evidence**

Capture:

- targeted and full pytest counts;
- Ruff results;
- browser-test results;
- export manifest status for all phases;
- final `git status --short`;
- confirmation that `notebooks/03_A3_Phase.ipynb` was never committed.
