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
SENTENCE_DASH_RE = re.compile(r"(?:—|–|(?<=\S)\s+--?\s+(?=\S))")
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
GERMAN_HEADING_TERM_RE = re.compile(
    r"\b(?:"
    r"Stufe|Position im|Systematischer|Fehleranalyse|Formale KPI-Validierung|"
    r"Bewertungsmatrix|Erklärbarkeit|Abgleich|Limitationen|Modellentscheidung|"
    r"Übergabe|Zusammenfassung"
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
SOURCE_TABLE_FIELDS = {
    "citation",
    "dataset",
    "documentation",
    "doi",
    "license",
    "publisher",
    "reference",
    "source",
    "url",
}
PRESENTATION_CALL_NAMES = {
    "Markdown",
    "add_annotation",
    "add_hline",
    "add_vline",
    "display",
    "make_subplots",
    "plot_binary_f1_recall_front",
    "plot_confusion_matrix_heatmap",
    "plot_f1_recall_front",
    "plot_roc_pr_curves",
    "print",
    "update_layout",
    "update_traces",
    "update_xaxes",
    "update_yaxes",
}
PRESENTATION_KEYWORDS = {
    "annotation_text",
    "colorbar",
    "hovertemplate",
    "labels",
    "name",
    "subplot_titles",
    "text",
    "ticktext",
    "title",
    "title_prefix",
    "title_text",
    "xaxis_title",
    "yaxis_title",
}


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
    prose_lines: list[str] = []
    in_fence = False
    for line in source.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [re.sub(r"[*_]", "", cell).strip() for cell in stripped.strip("|").split("|")]
            if cells and cells[0].casefold().rstrip(":") in SOURCE_TABLE_FIELDS:
                continue
            prose_lines.append(line)
            prose_lines.append("")
            continue

        if "http://" in line or "https://" in line or re.search(r"\bdoi\s*:", line, re.IGNORECASE):
            line = re.sub(r"\[[^\]]+\]\([^)]+\)", "", line)
            if not line.strip(" -*>"):
                continue
        prose_lines.append(line)

    without_inline_code = re.sub(r"`[^`]*`", "", "\n".join(prose_lines))
    return [block.strip() for block in re.split(r"\n\s*\n", without_inline_code) if block.strip()]


def call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{call_name(node.value)}.{node.attr}"
    return ""


def string_literals(node: ast.AST) -> list[str]:
    return [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


def code_presentation_literals(source: str) -> list[str]:
    tree = ast.parse(source)
    literals = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        qualified_name = call_name(node.func)
        short_name = qualified_name.rsplit(".", maxsplit=1)[-1]
        is_plotly_constructor = qualified_name.startswith(("go.", "px."))
        if short_name not in PRESENTATION_CALL_NAMES and not is_plotly_constructor:
            continue

        if short_name in {"print", "display", "Markdown"}:
            for argument in node.args:
                literals.extend(string_literals(argument))

        for keyword in node.keywords:
            if keyword.arg in PRESENTATION_KEYWORDS:
                literals.extend(string_literals(keyword.value))
    return literals


def german_presentation_markers(text: str, *, heading: bool = False) -> list[str]:
    explicit_terms = GERMAN_HEADING_TERM_RE.findall(text) if heading else []
    function_words = GERMAN_FUNCTION_WORD_RE.findall(text)
    if explicit_terms or len(function_words) >= 2:
        return sorted({*(term.lower() for term in explicit_terms), *function_words})
    return []


def german_code_presentation_markers(text: str) -> list[str]:
    explicit_terms = GERMAN_PRESENTATION_TERM_RE.findall(text)
    function_words = GERMAN_FUNCTION_WORD_RE.findall(text)
    if explicit_terms or len(function_words) >= 2:
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


def test_no_sentence_level_dash_punctuation_in_markdown():
    violations = []

    for path in NOTEBOOKS:
        notebook = read_notebook(path)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "markdown":
                continue
            for block in markdown_prose_blocks(cell.source):
                prose = re.sub(r"(?m)^\s*[-*+]\s+", "", block)
                if SENTENCE_DASH_RE.search(prose):
                    excerpt = " ".join(block.split())[:140]
                    violations.append(f"{path}: markdown cell {index}: {excerpt!r}")

    assert not violations, "\n".join(violations[:30])


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
                marker_finder = lambda text: german_presentation_markers(  # noqa: E731
                    text, heading=text.lstrip().startswith("#")
                )
            elif cell.cell_type == "code":
                presentation_strings = code_presentation_literals(cell.source)
                marker_finder = german_code_presentation_markers
            else:
                continue

            for text in presentation_strings:
                matched = marker_finder(text)
                if matched:
                    excerpt = " ".join(text.split())[:120]
                    violations.append(
                        f"{path}: {cell.cell_type} cell {index} contains {matched}: {excerpt!r}"
                    )

    assert not violations, "\n".join(violations[:30])


def test_language_contract_allows_source_metadata_raw_labels_and_internal_docstrings():
    markdown = """
| Item | Value |
|:---|:---|
| **Dataset** | Unfallatlas Deutschland |
| **Publisher** | Statistisches Bundesamt (Destatis) |

[Straßenverkehrsunfälle in Deutschland](https://example.invalid/source)

`UKATGEORIE=1` is the raw label `Getötet`.
"""
    assert not [
        block
        for block in markdown_prose_blocks(markdown)
        if german_presentation_markers(block, heading=block.lstrip().startswith("#"))
    ]

    source = '''
def helper():
    """Internal Stufe description; die Werte werden nicht displayed."""
    return "Wochentag"
'''
    assert code_presentation_literals(source) == []


def test_language_contract_detects_headings_body_prose_and_chart_labels():
    assert german_presentation_markers("## 3 — Stufe 0: baselines", heading=True)
    assert german_presentation_markers(
        "Die Fehler werden nach Unfalltyp und Straßenzustand verglichen."
    )

    source = """
fig.update_layout(title="Höchste False-Negative-Raten nach Slice")
"""
    literals = code_presentation_literals(source)
    assert literals == ["Höchste False-Negative-Raten nach Slice"]
    assert german_code_presentation_markers(literals[0])


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
