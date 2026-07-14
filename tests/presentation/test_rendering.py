from pathlib import Path

import nbformat
from bs4 import BeautifulSoup
from nbconvert import HTMLExporter

from unfallatlas.presentation import models

PACKAGE_ROOT = Path(__file__).parents[2] / "src" / "unfallatlas" / "presentation"
TEMPLATE_ROOT = PACKAGE_ROOT / "templates"
STATIC_ROOT = PACKAGE_ROOT / "static"


def _notebook() -> nbformat.NotebookNode:
    plotly = nbformat.v4.new_output(
        "display_data",
        data={"text/plain": "Figure"},
        metadata={
            "unfallatlas_presentation": {
                "kind": "plotly",
                "chart_id": "example-cell-2-output-0",
                "payload_key": "plotly-payload-digest",
                "asset_href": "../assets/notebooks/example/plotly-digest.js",
                "size_bytes": 418,
            }
        },
    )
    image = nbformat.v4.new_output(
        "display_data",
        data={"text/plain": "Image"},
        metadata={
            "unfallatlas_presentation": {
                "kind": "image",
                "asset_href": "../assets/notebooks/example/image-digest.png",
                "size_bytes": 117,
            }
        },
    )
    table = nbformat.v4.new_output(
        "display_data",
        data={"text/html": "<table><thead><tr><th>Klasse</th></tr></thead></table>"},
    )
    active_html = nbformat.v4.new_output(
        "display_data",
        data={"text/html": "<script>document.body.textContent = 'lokal'</script>"},
    )
    error = nbformat.v4.new_output(
        "error",
        ename="ValueError",
        evalue="Beispiel",
        traceback=["ValueError: Beispiel"],
    )
    return nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Größen *Qualität*\nEin kurzer Befund."),
            nbformat.v4.new_markdown_cell("## Größen `Qualität`\nVertiefung."),
            nbformat.v4.new_code_cell(
                "value = 1",
                execution_count=1,
                outputs=[
                    plotly,
                    image,
                    table,
                    active_html,
                    nbformat.v4.new_output("stream", name="stdout", text="Ausgabe\n"),
                    nbformat.v4.new_output("stream", name="stderr", text="Hinweis\n"),
                    error,
                ],
            ),
        ]
    )


def _presentation_resources(toc: tuple[object, ...]) -> dict[str, object]:
    return {
        "title": "Unfallatlas Analyse",
        "status": "ready",
        "source_path": "notebooks/example.ipynb",
        "snapshot_sha256": "a" * 64,
        "toc": toc,
        "findings": [
            {
                "code": "LARGE_OUTPUT",
                "severity": "warning",
                "message": "Große Ausgabe",
            }
        ],
        "counts": {
            "markdown": 2,
            "code": 1,
            "raw": 0,
            "unexecuted_code": 0,
            "executed_without_output": 0,
            "code_with_output": 1,
            "error_outputs": 1,
        },
        "metadata": {
            "exported_at_local": "15.07.2026, 10:30:00 CEST",
            "git": {
                "commit": "abc123def456",
                "short_commit": "abc123d",
                "branch": "feature/presentation",
                "dirty": True,
            },
        },
        "style_href": "../assets/ui/presentation.css",
        "script_href": "../assets/ui/presentation.js",
        "plotly_runtime_href": "../assets/vendor/plotly-6.1.0.min.js",
        "mathjax_runtime_href": "../assets/vendor/mathjax-3.2.2-tex-svg-full.js",
    }


def _render() -> tuple[str, nbformat.NotebookNode]:
    notebook = models.add_stable_heading_anchors(_notebook())
    toc = models.build_toc(notebook)
    exporter = HTMLExporter(
        template_name="notebook",
        extra_template_basedirs=[str(TEMPLATE_ROOT)],
    )
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    html, _ = exporter.from_notebook_node(
        notebook,
        resources={"presentation": _presentation_resources(toc)},
    )
    return html, notebook


def test_heading_helpers_create_stable_unicode_anchors_without_mutation() -> None:
    source = _notebook()

    anchored = models.add_stable_heading_anchors(source)
    toc = models.build_toc(anchored)

    assert anchored is not source
    assert source.cells[0].source.startswith("# Größen *Qualität*")
    assert [entry.level for entry in toc] == [1, 2]
    assert [entry.title for entry in toc] == ["Größen Qualität", "Größen Qualität"]
    assert [entry.anchor for entry in toc] == ["grossen-qualitat", "grossen-qualitat-2"]
    assert anchored.cells[0].source == (
        '<span id="grossen-qualitat" class="heading-anchor" aria-hidden="true"></span>\n'
        "# Größen *Qualität*\nEin kurzer Befund."
    )


def test_custom_template_renders_semantic_controls_metadata_and_outputs() -> None:
    html, _ = _render()
    soup = BeautifulSoup(html, "html.parser")

    assert soup.select_one("header.presentation-header")
    assert soup.select_one('nav[aria-label="Inhaltsverzeichnis"]')
    assert [item["href"] for item in soup.select(".toc-link")] == [
        "#grossen-qualitat",
        "#grossen-qualitat-2",
    ]
    assert soup.select_one("details.code-cell:not([open])")
    assert soup.select_one("details.output-cell[open]")
    assert {button["data-action"] for button in soup.select("button[data-action]")} >= {
        "show-all-code",
        "hide-all-code",
        "show-all-output",
        "hide-all-output",
        "print",
    }
    assert soup.select_one('[aria-live="polite"]')
    assert "https://cdn" not in html

    plotly = soup.select_one(".plotly-output[data-asset]")
    assert plotly
    assert plotly["data-payload-key"] == "plotly-payload-digest"
    assert soup.select_one('.output-image img[src$="image-digest.png"]')
    assert soup.select_one(".table-scroll table")
    assert soup.select_one(".sandboxed-output[sandbox][srcdoc]")
    assert soup.select_one(".text-output")
    assert soup.select_one(".error-output")

    metadata = soup.select_one(".metadata-grid")
    assert metadata
    metadata_text = metadata.get_text(" ", strip=True)
    assert "abc123d" in metadata_text
    assert "Arbeitsbaum geändert" in metadata_text
    assert "15.07.2026, 10:30:00 CEST" in metadata_text
    assert "2 Markdown" in metadata_text
    assert "1 Code" in metadata_text
    assert "1 Fehlerausgabe" in metadata_text
    assert "Große Ausgabe" in soup.get_text(" ", strip=True)


def test_template_preserves_markdown_and_pygments_code_markup() -> None:
    html, anchored = _render()
    soup = BeautifulSoup(html, "html.parser")

    assert soup.select_one(".notebook-main h1 em")
    assert soup.select_one(".notebook-main h2 code")
    assert soup.select_one(".code-cell .highlight pre")
    assert "value = 1" in soup.select_one(".code-cell").get_text(" ", strip=True)
    assert anchored.cells[0].source.endswith("# Größen *Qualität*\nEin kurzer Befund.")


def test_progressive_enhancement_and_print_styles_are_packaged_offline() -> None:
    javascript = (STATIC_ROOT / "presentation.js").read_text(encoding="utf-8")
    stylesheet = (STATIC_ROOT / "presentation.css").read_text(encoding="utf-8")

    assert "window.UnfallatlasPresentation" in javascript
    assert "registerPlotlyPayload" in javascript
    assert "waitForPayload(container.dataset.payloadKey)" in javascript
    assert "const plotlyLoads = new Map()" in javascript
    assert "return plotlyLoads.get(container)" in javascript
    assert "sessionStorage" in javascript
    assert "data-snapshot-sha256" in javascript
    assert "IntersectionObserver" in javascript
    assert "beforeprint" in javascript
    assert "window.print()" in javascript
    assert "https://" not in javascript

    for selector in (
        ".presentation-shell",
        ".presentation-header",
        ".metadata-grid",
        ".toc",
        ".notebook-main",
        ".code-cell",
        ".output-cell",
        ".table-scroll",
        ".text-output",
        ".error-output",
        ".plotly-output",
        ".focus-visible",
        ".back-to-top",
    ):
        assert selector in stylesheet
    assert "@media print" in stylesheet
    assert "@media (prefers-reduced-motion: reduce)" in stylesheet
    assert "max-height: 32rem" in stylesheet
    assert "max-height: 28rem" in stylesheet
    assert "https://" not in stylesheet
