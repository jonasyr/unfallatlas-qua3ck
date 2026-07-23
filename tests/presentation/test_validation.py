from pathlib import Path

import nbformat
import pytest
from conftest import write_notebook

from unfallatlas.presentation.metadata import snapshot_sha256
from unfallatlas.presentation.models import NotebookStatus, Severity
from unfallatlas.presentation.validation import (
    LARGE_OUTPUT_BYTES,
    VERY_LARGE_NOTEBOOK_OUTPUT_BYTES,
    NotebookReadError,
    read_and_validate_notebook,
    scan_runtime_references,
)


def codes(analysis) -> set[str]:
    return {finding.code for finding in analysis.findings}


def display_data(data: dict[str, object]):
    return nbformat.v4.new_output("display_data", data=data)


def test_size_thresholds_are_five_and_one_hundred_mib() -> None:
    assert LARGE_OUTPUT_BYTES == 5 * 1024 * 1024
    assert VERY_LARGE_NOTEBOOK_OUTPUT_BYTES == 100 * 1024 * 1024


def test_validate_rendered_html_detects_literal_markdown_table() -> None:
    from unfallatlas.presentation import validation

    html = "<main><p>| A | B |\n|---|---|\n| 1 | 2 |</p></main>"

    findings = validation.validate_rendered_html(html)

    assert [finding.code for finding in findings] == ["literal-markdown-table"]
    assert findings[0].severity is Severity.ERROR


def test_validate_rendered_html_ignores_prose_with_a_pipe() -> None:
    from unfallatlas.presentation import validation

    findings = validation.validate_rendered_html("<main><p>A | B is ordinary prose.</p></main>")

    assert findings == ()


def test_markdown_only_notebook_is_ready(tmp_path: Path) -> None:
    path = write_notebook(
        tmp_path / "notebooks/intro.ipynb", [nbformat.v4.new_markdown_cell("# Intro")]
    )

    analysis = read_and_validate_notebook(path, tmp_path)

    assert analysis.status is NotebookStatus.READY
    assert analysis.title == "Intro"
    assert analysis.counts.markdown == 1
    assert analysis.counts.code == 0
    assert analysis.findings == ()


def test_executed_text_output_is_counted(tmp_path: Path) -> None:
    output = nbformat.v4.new_output("stream", name="stdout", text="done\n")
    cell = nbformat.v4.new_code_cell("print('done')", execution_count=1, outputs=[output])
    path = write_notebook(tmp_path / "notebooks/run.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    assert analysis.counts.code_with_output == 1
    assert analysis.counts.executed_without_output == 0
    assert analysis.output_bytes > 0
    assert not analysis.strict_blocked


def test_empty_code_cell_is_not_counted_as_unexecuted(tmp_path: Path) -> None:
    path = write_notebook(tmp_path / "notebooks/empty.ipynb", [nbformat.v4.new_code_cell("  \n")])

    analysis = read_and_validate_notebook(path, tmp_path)

    assert analysis.counts.code == 0
    assert "UNEXECUTED_CELL" not in codes(analysis)


def test_unexecuted_code_cell_is_strict_wip(tmp_path: Path) -> None:
    path = write_notebook(tmp_path / "notebooks/wip.ipynb", [nbformat.v4.new_code_cell("x = 1")])

    analysis = read_and_validate_notebook(path, tmp_path)

    assert analysis.status is NotebookStatus.WIP
    assert analysis.counts.unexecuted_code == 1
    assert {"UNEXECUTED_CELL", "WIP_NOTEBOOK"} <= codes(analysis)
    assert analysis.strict_blocked


def test_executed_outputless_cell_is_info_not_strict(tmp_path: Path) -> None:
    cell = nbformat.v4.new_code_cell("x = 1", execution_count=1)
    path = write_notebook(tmp_path / "notebooks/setup.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    finding = next(item for item in analysis.findings if item.code == "EXECUTED_NO_OUTPUT")
    assert finding.severity is Severity.INFO
    assert finding.strict_blocker is False
    assert analysis.counts.executed_without_output == 1


def test_error_output_is_strict(tmp_path: Path) -> None:
    output = nbformat.v4.new_output("error", ename="ValueError", evalue="bad", traceback=[])
    cell = nbformat.v4.new_code_cell("raise ValueError", execution_count=1, outputs=[output])
    path = write_notebook(tmp_path / "notebooks/error.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    finding = next(item for item in analysis.findings if item.code == "ERROR_OUTPUT")
    assert finding.severity is Severity.ERROR
    assert finding.cell_index == 0
    assert finding.strict_blocker
    assert analysis.counts.error_outputs == 1
    assert analysis.status is NotebookStatus.WIP
    assert "WIP_NOTEBOOK" in codes(analysis)


def test_increasing_execution_counts_are_valid(tmp_path: Path) -> None:
    cells = [
        nbformat.v4.new_code_cell("x = 1", execution_count=1),
        nbformat.v4.new_code_cell("x += 1", execution_count=3),
    ]
    path = write_notebook(tmp_path / "notebooks/ordered.ipynb", cells)

    analysis = read_and_validate_notebook(path, tmp_path)

    assert "NON_MONOTONIC_EXECUTION" not in codes(analysis)
    assert "DUPLICATE_EXECUTION_COUNT" not in codes(analysis)


def test_decreasing_execution_counts_are_strict(tmp_path: Path) -> None:
    cells = [
        nbformat.v4.new_code_cell("x = 1", execution_count=2),
        nbformat.v4.new_code_cell("x += 1", execution_count=1),
    ]
    path = write_notebook(tmp_path / "notebooks/unordered.ipynb", cells)

    analysis = read_and_validate_notebook(path, tmp_path)

    assert "NON_MONOTONIC_EXECUTION" in codes(analysis)
    assert analysis.strict_blocked


def test_duplicate_execution_counts_are_strict(tmp_path: Path) -> None:
    cells = [
        nbformat.v4.new_code_cell("x = 1", execution_count=1),
        nbformat.v4.new_code_cell("x += 1", execution_count=1),
    ]
    path = write_notebook(tmp_path / "notebooks/duplicate.ipynb", cells)

    analysis = read_and_validate_notebook(path, tmp_path)

    assert "DUPLICATE_EXECUTION_COUNT" in codes(analysis)
    assert analysis.strict_blocked


def test_supported_html_fallback_avoids_unsupported_mime(tmp_path: Path) -> None:
    output = display_data(
        {"application/x-custom": "custom", "text/html": "<strong>fallback</strong>"}
    )
    cell = nbformat.v4.new_code_cell("show()", execution_count=1, outputs=[output])
    path = write_notebook(tmp_path / "notebooks/fallback.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    assert "UNSUPPORTED_MIME" not in codes(analysis)


def test_unsupported_only_mime_is_strict(tmp_path: Path) -> None:
    output = display_data({"application/x-custom": "custom"})
    cell = nbformat.v4.new_code_cell("show()", execution_count=1, outputs=[output])
    path = write_notebook(tmp_path / "notebooks/unsupported.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    assert "UNSUPPORTED_MIME" in codes(analysis)
    assert analysis.strict_blocked


def test_data_wrangler_requires_fallback(tmp_path: Path) -> None:
    mime = "application/vnd.dataresource+json"
    unsupported = display_data({mime: {"data": []}})
    supported = display_data({mime: {"data": []}, "text/plain": "empty table"})
    cells = [
        nbformat.v4.new_code_cell("first", execution_count=1, outputs=[unsupported]),
        nbformat.v4.new_code_cell("second", execution_count=2, outputs=[supported]),
    ]
    path = write_notebook(tmp_path / "notebooks/wrangler.ipynb", cells)

    analysis = read_and_validate_notebook(path, tmp_path)

    unsupported_findings = [item for item in analysis.findings if item.code == "UNSUPPORTED_MIME"]
    assert [item.cell_index for item in unsupported_findings] == [0]


def test_widget_without_state_is_strict(tmp_path: Path) -> None:
    output = display_data(
        {
            "application/vnd.jupyter.widget-view+json": {
                "model_id": "widget-id",
                "version_major": 2,
                "version_minor": 0,
            }
        }
    )
    cell = nbformat.v4.new_code_cell("widget", execution_count=1, outputs=[output])
    path = write_notebook(tmp_path / "notebooks/widget.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    assert "WIDGET_STATE_MISSING" in codes(analysis)
    assert "WIDGET_UNSUPPORTED" in codes(analysis)
    assert "UNSUPPORTED_MIME" in codes(analysis)
    assert analysis.strict_blocked


def test_widget_without_state_uses_valid_html_fallback(tmp_path: Path) -> None:
    output = display_data(
        {
            "application/vnd.jupyter.widget-view+json": {"model_id": "missing-widget-id"},
            "text/html": "<strong>static widget fallback</strong>",
        }
    )
    cell = nbformat.v4.new_code_cell("widget", execution_count=1, outputs=[output])
    path = write_notebook(tmp_path / "notebooks/widget-fallback.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    assert "WIDGET_STATE_MISSING" not in codes(analysis)
    finding = next(item for item in analysis.findings if item.code == "WIDGET_UNSUPPORTED")
    assert finding.severity is Severity.WARNING
    assert not finding.strict_blocker
    assert not analysis.strict_blocked
    assert analysis.notebook.cells[0].outputs[0].data == output.data
    assert analysis.snapshot_sha256 == snapshot_sha256(analysis.notebook)


def test_widget_with_notebook_state_still_requires_static_fallback(tmp_path: Path) -> None:
    output = display_data(
        {
            "application/vnd.jupyter.widget-view+json": {"model_id": "widget-id"},
            "text/plain": "static widget fallback",
        }
    )
    metadata = {
        "widgets": {
            "application/vnd.jupyter.widget-state+json": {
                "state": {"widget-id": {}},
                "version_major": 2,
                "version_minor": 0,
            }
        }
    }
    cell = nbformat.v4.new_code_cell("widget", execution_count=1, outputs=[output])
    path = write_notebook(tmp_path / "notebooks/widget.ipynb", [cell], metadata)

    analysis = read_and_validate_notebook(path, tmp_path)

    assert "WIDGET_STATE_MISSING" not in codes(analysis)
    assert "WIDGET_UNSUPPORTED" in codes(analysis)
    assert "UNSUPPORTED_MIME" not in codes(analysis)
    assert not analysis.strict_blocked
    assert analysis.notebook.cells[0].outputs[0].data == output.data


def test_widget_with_state_but_no_static_fallback_is_strict(tmp_path: Path) -> None:
    output = display_data({"application/vnd.jupyter.widget-view+json": {"model_id": "widget-id"}})
    metadata = {
        "widgets": {
            "application/vnd.jupyter.widget-state+json": {
                "state": {"widget-id": {}},
                "version_major": 2,
                "version_minor": 0,
            }
        }
    }
    cell = nbformat.v4.new_code_cell("widget", execution_count=1, outputs=[output])
    path = write_notebook(tmp_path / "notebooks/widget-only.ipynb", [cell], metadata)

    analysis = read_and_validate_notebook(path, tmp_path)

    assert "WIDGET_STATE_MISSING" not in codes(analysis)
    assert "WIDGET_UNSUPPORTED" in codes(analysis)
    assert "UNSUPPORTED_MIME" in codes(analysis)
    assert analysis.strict_blocked
    assert analysis.notebook.cells[0].outputs[0].data == output.data


def test_widget_javascript_is_not_accepted_as_static_fallback(tmp_path: Path) -> None:
    output = display_data(
        {
            "application/vnd.jupyter.widget-view+json": {"model_id": "widget-id"},
            "application/javascript": "window.alert('active')",
        }
    )
    cell = nbformat.v4.new_code_cell("widget", execution_count=1, outputs=[output])
    path = write_notebook(tmp_path / "notebooks/widget-script.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    finding = next(item for item in analysis.findings if item.code == "WIDGET_UNSUPPORTED")
    assert finding.severity is Severity.ERROR
    assert finding.strict_blocker
    assert analysis.notebook.cells[0].outputs[0].data == output.data


@pytest.mark.parametrize(
    "html",
    [
        "<script>window.renderWidget()</script>",
        '<button onclick="window.renderWidget()">Render</button>',
        "<div></div>",
    ],
)
def test_widget_html_fallback_must_be_visible_and_static(tmp_path: Path, html: str) -> None:
    output = display_data(
        {
            "application/vnd.jupyter.widget-view+json": {"model_id": "widget-id"},
            "text/html": html,
        }
    )
    cell = nbformat.v4.new_code_cell("widget", execution_count=1, outputs=[output])
    path = write_notebook(tmp_path / "notebooks/widget-html.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    finding = next(item for item in analysis.findings if item.code == "WIDGET_UNSUPPORTED")
    assert finding.severity is Severity.ERROR
    assert finding.strict_blocker
    assert analysis.strict_blocked


def test_present_local_markdown_image_is_allowed(tmp_path: Path) -> None:
    image = tmp_path / "notebooks/images/chart.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    cell = nbformat.v4.new_markdown_cell("![chart](images/chart.png)")
    path = write_notebook(tmp_path / "notebooks/present.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    assert "MISSING_LOCAL_ASSET" not in codes(analysis)
    assert "UNSAFE_LOCAL_ASSET" not in codes(analysis)


def test_missing_local_image_blocks_strict(tmp_path: Path) -> None:
    cell = nbformat.v4.new_markdown_cell("![chart](missing.png)")
    path = write_notebook(tmp_path / "notebooks/missing.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    assert "MISSING_LOCAL_ASSET" in codes(analysis)
    assert analysis.strict_blocked


def test_traversal_outside_repo_is_unsafe_even_when_file_exists(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"secret")
    cell = nbformat.v4.new_markdown_cell("![secret](../../secret.png)")
    path = write_notebook(repo / "notebooks/traversal.ipynb", [cell])

    analysis = read_and_validate_notebook(path, repo)

    assert "UNSAFE_LOCAL_ASSET" in codes(analysis)
    assert "MISSING_LOCAL_ASSET" not in codes(analysis)
    assert analysis.strict_blocked


@pytest.mark.parametrize(
    "html",
    [
        '<img src="https://example.org/chart.png">',
        '<script src="//cdn.example.org/app.js"></script>',
        '<iframe src="https://example.org/embed"></iframe>',
        '<style>.hero { background: url("https://example.org/hero.png") }</style>',
    ],
)
def test_external_runtime_resources_are_strict(tmp_path: Path, html: str) -> None:
    findings = scan_runtime_references(html, tmp_path, tmp_path, 4)

    finding = next(item for item in findings if item.code == "EXTERNAL_RUNTIME_RESOURCE")
    assert finding.cell_index == 4
    assert finding.strict_blocker


def test_raw_html_in_markdown_is_scanned(tmp_path: Path) -> None:
    cell = nbformat.v4.new_markdown_cell('<video src="https://example.org/movie.mp4"></video>')
    path = write_notebook(tmp_path / "notebooks/video.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    assert "EXTERNAL_RUNTIME_RESOURCE" in codes(analysis)


def test_normal_external_hyperlink_is_not_runtime_dependency(tmp_path: Path) -> None:
    cell = nbformat.v4.new_markdown_cell("[Source](https://example.org/paper)")
    path = write_notebook(tmp_path / "notebooks/link.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    assert "EXTERNAL_RUNTIME_RESOURCE" not in codes(analysis)


def test_fragment_mailto_and_embedded_data_are_ignored(tmp_path: Path) -> None:
    html = (
        '<img src="data:image/png;base64,AAAA"><a href="#part">x</a><a href="mailto:a@b.de">m</a>'
    )

    assert scan_runtime_references(html, tmp_path, tmp_path, 0) == ()


def test_plotly_open_street_map_tiles_are_strict(tmp_path: Path) -> None:
    plotly = {"data": [], "layout": {"mapbox": {"style": "open-street-map"}}}
    output = display_data({"application/vnd.plotly.v1+json": plotly})
    cell = nbformat.v4.new_code_cell("figure", execution_count=1, outputs=[output])
    path = write_notebook(tmp_path / "notebooks/map.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    assert "EXTERNAL_MAP_TILES" in codes(analysis)
    assert analysis.strict_blocked


@pytest.mark.parametrize(
    "layout",
    [
        {"mapbox2": {"style": "open-street-map"}},
        {"map2": {"style": "open-street-map"}},
        {
            "mapbox": {
                "style": {
                    "version": 8,
                    "sources": {"osm": {"tiles": ["https://tiles.example.org/{z}/{x}/{y}.png"]}},
                }
            }
        },
        {"mapbox2": {"layers": {"roads": {"source": "https://tiles.example.org/roads"}}}},
    ],
)
def test_plotly_numbered_and_mapping_maps_require_external_tiles(
    tmp_path: Path, layout: dict[str, object]
) -> None:
    output = display_data({"application/vnd.plotly.v1+json": {"data": [], "layout": layout}})
    cell = nbformat.v4.new_code_cell("figure", execution_count=1, outputs=[output])
    path = write_notebook(tmp_path / "notebooks/map-subplot.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    assert "EXTERNAL_MAP_TILES" in codes(analysis)
    assert analysis.strict_blocked


def test_plotly_external_layout_image_is_runtime_resource(tmp_path: Path) -> None:
    plotly = {
        "data": [],
        "layout": {"images": [{"source": "https://example.org/watermark.png"}]},
    }
    output = display_data({"application/vnd.plotly.v1+json": plotly})
    cell = nbformat.v4.new_code_cell("figure", execution_count=1, outputs=[output])
    path = write_notebook(tmp_path / "notebooks/image.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    assert "EXTERNAL_RUNTIME_RESOURCE" in codes(analysis)
    assert analysis.strict_blocked


def test_plotly_missing_local_layout_image_is_structured_finding(tmp_path: Path) -> None:
    plotly = {
        "data": [],
        "layout": {"images": [{"source": "images/missing-watermark.png"}]},
    }
    output = display_data({"application/vnd.plotly.v1+json": plotly})
    cell = nbformat.v4.new_code_cell("figure", execution_count=1, outputs=[output])
    path = write_notebook(tmp_path / "notebooks/image.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    assert "MISSING_LOCAL_ASSET" in codes(analysis)
    assert analysis.strict_blocked


def test_single_payload_over_five_mib_warns_but_does_not_block(tmp_path: Path) -> None:
    text = "x" * LARGE_OUTPUT_BYTES
    output = display_data({"text/plain": text})
    cell = nbformat.v4.new_code_cell("large", execution_count=1, outputs=[output])
    path = write_notebook(tmp_path / "notebooks/large.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    finding = next(item for item in analysis.findings if item.code == "LARGE_OUTPUT")
    assert finding.severity is Severity.WARNING
    assert not finding.strict_blocker


def test_total_payload_over_one_hundred_mib_warns_but_does_not_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("unfallatlas.presentation.validation.VERY_LARGE_NOTEBOOK_OUTPUT_BYTES", 100)
    outputs = [display_data({"text/plain": "x" * 60}) for _ in range(2)]
    cell = nbformat.v4.new_code_cell("many", execution_count=1, outputs=outputs)
    path = write_notebook(tmp_path / "notebooks/very-large.ipynb", [cell])

    analysis = read_and_validate_notebook(path, tmp_path)

    finding = next(item for item in analysis.findings if item.code == "VERY_LARGE_NOTEBOOK_OUTPUT")
    assert analysis.output_bytes > 100
    assert finding.severity is Severity.WARNING
    assert not finding.strict_blocker


def test_placeholder_status_emits_warning_not_strict(tmp_path: Path) -> None:
    path = write_notebook(
        tmp_path / "notebooks/todo.ipynb", [nbformat.v4.new_markdown_cell("# TODO")]
    )

    analysis = read_and_validate_notebook(path, tmp_path)

    assert analysis.status is NotebookStatus.PLACEHOLDER
    finding = next(item for item in analysis.findings if item.code == "PLACEHOLDER_NOTEBOOK")
    assert finding.severity is Severity.WARNING
    assert not finding.strict_blocker
    assert not analysis.strict_blocked


@pytest.mark.parametrize("content", ["not json", '{"nbformat": 4}'])
def test_invalid_notebook_is_wrapped_with_source_path(tmp_path: Path, content: str) -> None:
    path = tmp_path / "notebooks/broken.ipynb"
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")

    with pytest.raises(NotebookReadError, match="broken.ipynb"):
        read_and_validate_notebook(path, tmp_path)
