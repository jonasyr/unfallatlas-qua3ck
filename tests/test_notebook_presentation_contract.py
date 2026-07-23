from pathlib import Path

import nbformat

NOTEBOOKS = [
    Path("notebooks/01_Q_Phase.ipynb"),
    Path("notebooks/02_U_Phase.ipynb"),
    Path("notebooks/03_A3_Phase.ipynb"),
    Path("notebooks/04_C_Phase.ipynb"),
]

PLOTLY_MIME = "application/vnd.plotly.v1+json"
CHART_CALL_MARKERS = (
    "go.Figure(",
    "make_subplots(",
    "plot_binary_f1_recall_front(",
    "plot_confusion_matrix_heatmap(",
    "plot_f1_recall_front(",
    "plot_roc_pr_curves(",
    "px.",
)


def read_notebook(path: Path) -> nbformat.NotebookNode:
    return nbformat.read(path, as_version=4)


def notebook_text(path: Path, *, cell_type: str | None = None) -> str:
    notebook = read_notebook(path)
    return "\n".join(
        "".join(cell.source)
        for cell in notebook.cells
        if cell_type is None or cell.cell_type == cell_type
    )


def test_no_matplotlib_or_seaborn_chart_code():
    forbidden = ("matplotlib", "seaborn", "plt.", "sns.")
    violations = []

    for path in NOTEBOOKS:
        notebook = read_notebook(path)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            matched = [token for token in forbidden if token in cell.source]
            if matched:
                violations.append(f"{path}: code cell {index} contains {matched}")

    assert not violations, "\n".join(violations)


def test_no_development_checklists():
    violations = []

    for path in NOTEBOOKS:
        markdown = notebook_text(path, cell_type="markdown")
        if "acceptance checklist" in markdown.lower() or "[ ]" in markdown:
            violations.append(str(path))

    assert not violations, f"Development checklists remain in: {', '.join(violations)}"


def test_no_stale_c_phase_artifact_limitation():
    c_markdown = notebook_text(Path("notebooks/04_C_Phase.ipynb"), cell_type="markdown").lower()
    stale_claims = (
        "only the champion pipeline",
        "pipelines were not persisted",
        "wurden nicht persistiert",
        "nur die champion-pipeline ist als artefakt gespeichert",
    )
    matched = [claim for claim in stale_claims if claim in c_markdown]

    assert not matched, f"Stale C-phase artifact limitations remain: {matched}"


def test_c_phase_presentation_narrative_is_english():
    c_markdown = notebook_text(Path("notebooks/04_C_Phase.ipynb"), cell_type="markdown")
    german_section_markers = (
        "Position im QUA³CK-Prozess",
        "Systematischer Modellvergleich",
        "Fehleranalyse nach Slices",
        "Formale KPI-Validierung",
        "Qualitative Bewertungsmatrix",
        "SHAP-Erklärbarkeit",
        "Abgleich mit der Literatur",
        "Limitationen",
        "Finale Modellentscheidung",
        "Übergabe an die K-Phase",
        "Zusammenfassung der C-Phase",
    )
    matched = [marker for marker in german_section_markers if marker in c_markdown]

    assert not matched, f"German presentation headings remain in C phase: {matched}"


def test_a3_multiclass_decision_precedes_binary_search():
    text = notebook_text(Path("notebooks/03_A3_Phase.ipynb"), cell_type="markdown")
    multiclass_heading = "Three-class feasibility decision"
    binary_heading = "Binary KSI candidate search"

    assert multiclass_heading in text, f"Missing A³ heading: {multiclass_heading}"
    assert binary_heading in text, f"Missing A³ heading: {binary_heading}"
    assert text.index(multiclass_heading) < text.index(binary_heading)
    assert text.count("A³ summary") == 1


def test_saved_chart_outputs_use_interactive_plotly_mime_not_static_png():
    violations = []

    for path in NOTEBOOKS:
        notebook = read_notebook(path)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code" or not any(
                marker in cell.source for marker in CHART_CALL_MARKERS
            ):
                continue

            output_mimes = {
                mime for output in cell.get("outputs", []) for mime in output.get("data", {}).keys()
            }
            if "image/png" in output_mimes:
                violations.append(f"{path}: chart code cell {index} saved static image/png output")
            elif output_mimes and PLOTLY_MIME not in output_mimes:
                violations.append(
                    f"{path}: chart code cell {index} has display output without {PLOTLY_MIME}"
                )

    assert not violations, "\n".join(violations)
