import ast
import re
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
MARKDOWN_TASK_BOX_RE = re.compile(r"(?m)^\s*(?:[-*+]\s+)?\[[ xX]\]\s+\S")
GERMAN_PRESENTATION_TERM_RE = re.compile(
    r"\b(?:"
    r"Stufe|Unfälle|Anzahl|Verkehrsmittel|Stunde|Wochentag|Schwere|tödlich|"
    r"niedriger|fehlend|Höchste|Limitationen|Modellentscheidung|Übergabe|"
    r"Zusammenfassung|Bewertungsmatrix|Erklärbarkeit|Fehleranalyse|"
    r"Modellvergleich|Basiswert|Vorhersage|Übrige|Einfluss|Richtung|"
    r"niedrig|hoch"
    r")\b",
    re.IGNORECASE,
)
GERMAN_FUNCTION_WORD_RE = re.compile(
    r"\b(?:"
    r"aber|als|auch|auf|aus|bei|das|dass|dem|den|der|des|die|dies|diese|"
    r"einem|einen|einer|eine|ein|für|gegenüber|im|ist|liegen|liegt|mit|"
    r"nicht|nur|oder|sind|über|um|und|unter|von|werden|wird|wurde|wurden|"
    r"zeigt|zeigen|zum|zur"
    r")\b",
    re.IGNORECASE,
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


def markdown_prose_blocks(source: str) -> list[str]:
    prose_lines = []
    in_fence = False
    for line in source.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            prose_lines.append(line)

    without_inline_code = re.sub(r"`[^`]*`", "", "\n".join(prose_lines))
    return [block.strip() for block in re.split(r"\n\s*\n", without_inline_code) if block.strip()]


def code_string_literals(source: str) -> list[str]:
    tree = ast.parse(source)
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def german_presentation_markers(text: str) -> list[str]:
    explicit_terms = GERMAN_PRESENTATION_TERM_RE.findall(text)
    function_words = GERMAN_FUNCTION_WORD_RE.findall(text)
    if explicit_terms or len(function_words) >= 3:
        return sorted({*(term.lower() for term in explicit_terms), *function_words})
    return []


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
        if "acceptance checklist" in markdown.lower() or MARKDOWN_TASK_BOX_RE.search(markdown):
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


def test_presentation_language_is_english():
    violations = []

    for path in NOTEBOOKS:
        notebook = read_notebook(path)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "markdown":
                presentation_strings = markdown_prose_blocks(cell.source)
            elif cell.cell_type == "code":
                presentation_strings = code_string_literals(cell.source)
            else:
                continue

            for text in presentation_strings:
                matched = german_presentation_markers(text)
                if matched:
                    excerpt = " ".join(text.split())[:120]
                    violations.append(
                        f"{path}: {cell.cell_type} cell {index} contains {matched}: {excerpt!r}"
                    )

    assert not violations, "\n".join(violations[:30])


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
                mime for output in cell.get("outputs", []) for mime in output.get("data", {})
            }
            if "image/png" in output_mimes:
                violations.append(f"{path}: chart code cell {index} saved static image/png output")
            if PLOTLY_MIME not in output_mimes:
                violations.append(
                    f"{path}: chart code cell {index} has no saved {PLOTLY_MIME} output"
                )

    assert not violations, "\n".join(violations)
