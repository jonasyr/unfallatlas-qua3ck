import copy
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import nbformat
import pytest
from bs4 import BeautifulSoup
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

from unfallatlas.presentation import models
from unfallatlas.presentation.models import (
    CellCounts,
    ExportMetadata,
    GitMetadata,
    NotebookAnalysis,
    NotebookStatus,
)

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


def _render(
    source_notebook: nbformat.NotebookNode | None = None,
) -> tuple[str, nbformat.NotebookNode]:
    notebook = models.add_stable_heading_anchors(source_notebook or _notebook())
    toc = models.build_toc(notebook)
    exporter = HTMLExporter(
        template_name="notebook",
        extra_template_basedirs=[str(TEMPLATE_ROOT)],
    )
    if hasattr(models, "classify_html_output"):
        exporter.register_filter("classify_html_output", models.classify_html_output)
    if hasattr(models, "nest_toc"):
        exporter.register_filter("nest_toc", models.nest_toc)
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    html, _ = exporter.from_notebook_node(
        notebook,
        resources={"presentation": _presentation_resources(toc)},
    )
    return html, notebook


def _html_output_notebook(values: list[str]) -> nbformat.NotebookNode:
    outputs = [
        nbformat.v4.new_output("display_data", data={"text/html": value}) for value in values
    ]
    return nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("display", execution_count=1, outputs=outputs)]
    )


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


def test_heading_anchors_are_unique_across_generated_and_literal_suffixes() -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_markdown_cell("# Foo\n# Foo\n# Foo-2")]
    )

    anchored = models.add_stable_heading_anchors(notebook)
    entries = models.build_toc(anchored)

    assert [entry.anchor for entry in entries] == ["foo", "foo-2", "foo-2-2"]
    assert len({entry.anchor for entry in entries}) == len(entries)


def test_heading_scanner_ignores_fences_and_preserves_meaningful_hashes() -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                """# Sichtbar
```python
# Nicht sichtbar
```
~~~text
## Auch nicht sichtbar
~~~
## C#
### Schluss ###
### Bedeutung###
"""
            )
        ]
    )

    anchored = models.add_stable_heading_anchors(notebook)
    entries = models.build_toc(anchored)

    assert [entry.title for entry in entries] == [
        "Sichtbar",
        "C#",
        "Schluss",
        "Bedeutung###",
    ]
    assert "<span" not in anchored.cells[0].source.split("```python", 1)[1].split("```", 1)[0]
    assert "<span" not in anchored.cells[0].source.split("~~~text", 1)[1].split("~~~", 1)[0]


@pytest.mark.parametrize(
    "value",
    [
        '<table><tr><td onclick="alert(1)">X</td></tr></table>',
        '<table><tr><td style="background:url(javascript:alert(1))">X</td></tr></table>',
        '<table><tr><td><a href="javascript:alert(1)">X</a></td></tr></table>',
        '<svg onload="alert(1)"><circle></circle></svg>',
        '<object data="payload.html"></object>',
        '<embed src="payload.html">',
        '<meta http-equiv="refresh" content="0;url=payload.html">',
        '<link rel="stylesheet" href="payload.css">',
        "<script>alert(1)</script>",
        '<iframe src="payload.html"></iframe>',
        "<p>Kein passiver Tabelleninhalt</p>",
    ],
)
def test_html_classifier_sandboxes_every_non_passive_table(value: str) -> None:
    classification = models.classify_html_output(value)

    assert classification.kind == "sandbox"
    assert classification.content == value


def test_html_classifier_allows_only_sanitized_passive_table_markup() -> None:
    value = (
        '<table class="data"><caption>Ergebnis</caption><thead><tr>'
        '<th scope="col" style="color:red">Klasse</th></tr></thead>'
        '<tbody><tr><td colspan="1" data-extra="x">A</td></tr></tbody></table>'
    )

    classification = models.classify_html_output(value)
    soup = BeautifulSoup(classification.content, "html.parser")

    assert classification.kind == "table"
    assert soup.select_one("table caption")
    assert soup.select_one('th[scope="col"]')
    assert soup.select_one('td[colspan="1"]')
    assert not soup.select_one("[class], [style], [data-extra]")


def test_template_sandboxes_active_html_and_renders_only_passive_table_inline() -> None:
    values = [
        '<table><tr><td scope="row">Sicher</td></tr></table>',
        '<table><tr><td onmouseover="alert(1)">Aktiv</td></tr></table>',
        "<svg><script>alert(1)</script></svg>",
        '<a href="javascript:alert(1)">Aktiv</a>',
    ]

    html, _ = _render(_html_output_notebook(values))
    soup = BeautifulSoup(html, "html.parser")

    assert len(soup.select(".table-scroll")) == 1
    assert len(soup.select(".sandboxed-output[sandbox][srcdoc]")) == 3
    assert not soup.select(".output-content [onmouseover], .output-content svg, .output-content a")


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


def test_toc_uses_semantically_nested_lists() -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                "# Erste\n## Unterpunkt\n### Detail\n## Zweiter Unterpunkt\n# Zweite"
            )
        ]
    )

    html, _ = _render(notebook)
    soup = BeautifulSoup(html, "html.parser")
    toc = soup.select_one('nav[aria-label="Inhaltsverzeichnis"]')

    top_items = toc.select(":scope > ol > li")
    assert len(top_items) == 2
    assert top_items[0].select_one(":scope > ol > li > a").get_text(strip=True) == "Unterpunkt"
    assert top_items[0].select_one(":scope > ol > li > ol > li > a").get_text(strip=True) == (
        "Detail"
    )


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
    assert "function readStorage" in javascript
    assert "function writeStorage" in javascript
    assert javascript.count("catch {") >= 2
    assert "data-snapshot-sha256" in javascript
    assert "IntersectionObserver" in javascript
    assert "beforeprint" in javascript
    assert "window.print()" in javascript
    assert 'matchMedia("(prefers-reduced-motion: reduce)")' in javascript
    assert 'behavior: reduceMotion.matches ? "auto" : "smooth"' in javascript
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
    assert ".toc-trigger, .toc-close" in stylesheet
    assert ".js .toc-trigger" in stylesheet
    assert ".js .toc-close" in stylesheet
    assert "https://" not in stylesheet


def test_toc_controls_are_exposed_only_for_enhanced_mobile_layout() -> None:
    html, _ = _render()
    soup = BeautifulSoup(html, "html.parser")
    trigger = soup.select_one('[data-action="toggle-toc"]')
    close = soup.select_one('[data-action="close-toc"]')
    stylesheet = (STATIC_ROOT / "presentation.css").read_text(encoding="utf-8")

    assert "toc-trigger" in trigger.get("class", [])
    assert "toc-close" in close.get("class", [])
    assert trigger["aria-expanded"] == "false"
    assert ".toc-heading button { display: none; }" not in stylesheet
    assert "  .toc-heading button { display: inline-block; }" not in stylesheet


def _renderer_analysis(tmp_path: Path) -> NotebookAnalysis:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Renderer Integration\nDer vollständige Export."),
            nbformat.v4.new_code_cell(
                "raise RuntimeError('must never execute')",
                execution_count=7,
                outputs=[
                    nbformat.v4.new_output("stream", name="stdout", text="Gespeicherter Text\n"),
                    nbformat.v4.new_output(
                        "display_data",
                        data={"text/html": "<table><tr><td>Gespeicherte Tabelle</td></tr></table>"},
                    ),
                    nbformat.v4.new_output(
                        "display_data",
                        data={
                            "image/svg+xml": "<svg xmlns='http://www.w3.org/2000/svg'></svg>",
                            "text/plain": "SVG fallback",
                        },
                    ),
                    nbformat.v4.new_output(
                        "display_data",
                        data={
                            "application/vnd.plotly.v1+json": {
                                "data": [{"type": "bar", "x": ["A"], "y": [1]}],
                                "layout": {"title": "Gespeichertes Plotly"},
                            },
                            "text/plain": "Plotly fallback",
                        },
                    ),
                ],
            ),
        ]
    )
    return NotebookAnalysis(
        source=tmp_path / "notebooks" / "renderer-integration.ipynb",
        notebook=notebook,
        title="Renderer Integration",
        status=NotebookStatus.READY,
        counts=CellCounts(1, 1, 0, 0, 0, 1, 0),
        findings=(),
        snapshot_sha256="a" * 64,
        source_sha256="b" * 64,
        output_bytes=100,
    )


def _renderer_metadata() -> ExportMetadata:
    return ExportMetadata(
        exported_at=datetime(2026, 7, 15, 8, 30, tzinfo=UTC),
        exported_at_local="2026-07-15T10:30:00+02:00",
        git=GitMetadata(
            commit="abc123def4567890",
            short_commit="abc123def456",
            branch="feature/presentation",
            dirty=False,
        ),
    )


def _fail_if_called(*args, **kwargs):
    del args, kwargs
    raise AssertionError("renderer must not execute notebooks or subprocesses")


def test_render_notebook_publishes_saved_outputs_and_local_assets_without_execution(
    tmp_path: Path, monkeypatch
) -> None:
    from unfallatlas.presentation.rendering import render_notebook

    analysis = _renderer_analysis(tmp_path)
    original = copy.deepcopy(analysis.notebook)
    output_root = tmp_path / "site"
    monkeypatch.setattr(subprocess, "run", _fail_if_called)
    monkeypatch.setattr(NotebookClient, "execute", _fail_if_called)
    monkeypatch.setattr(ExecutePreprocessor, "preprocess", _fail_if_called)

    result = render_notebook(analysis, _renderer_metadata(), output_root)

    assert result.error is None
    assert result.destination == output_root / "notebooks" / "renderer-integration.html"
    assert list((output_root / "notebooks").glob("*.html")) == [result.destination]
    assert result.size_bytes == result.destination.stat().st_size
    assert analysis.notebook == original
    assert analysis.notebook.cells[1].execution_count == 7
    assert {asset.kind for asset in result.assets} >= {
        "image",
        "plotly",
        "plotly-runtime",
        "mathjax-runtime",
        "ui-style",
        "ui-script",
    }
    assert all((output_root / asset.relative_path).is_file() for asset in result.assets)

    html = result.destination.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    assert soup.title.get_text(strip=True) == "Renderer Integration"
    assert "2026-07-15T10:30:00+02:00" in soup.get_text(" ", strip=True)
    assert "abc123def456" in soup.get_text(" ", strip=True)
    assert "Gespeicherter Text" in soup.get_text(" ", strip=True)
    assert "Gespeicherte Tabelle" in soup.get_text(" ", strip=True)
    plotly = soup.select_one(".plotly-output[data-payload-key][data-asset]")
    assert plotly["data-payload-key"].startswith("plotly-")

    local_references = {
        value
        for tag in soup.select("[href], [src], [data-asset]")
        for attribute in ("href", "src", "data-asset")
        if (value := tag.get(attribute)) and not value.startswith("#")
    }
    assert local_references
    assert all("://" not in reference for reference in local_references)
    assert all(
        (result.destination.parent / reference).resolve().is_file()
        for reference in local_references
    )


def test_render_notebook_preserves_previous_html_when_final_replace_fails(
    tmp_path: Path, monkeypatch
) -> None:
    import unfallatlas.presentation.rendering as rendering

    analysis = _renderer_analysis(tmp_path)
    output_root = tmp_path / "site"
    target = output_root / "notebooks" / "renderer-integration.html"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old html")
    original_replace = os.replace

    def fail_final_replace(source: Path, destination: Path) -> None:
        if Path(destination) == target:
            raise OSError("final HTML replace failed")
        original_replace(source, destination)

    monkeypatch.setattr(rendering.os, "replace", fail_final_replace)

    result = rendering.render_notebook(analysis, _renderer_metadata(), output_root)

    assert result.error == "final HTML replace failed"
    assert target.read_bytes() == b"old html"
    assert not [path for path in target.parent.iterdir() if path != target]
