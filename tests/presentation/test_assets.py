import base64
import copy
import hashlib
from pathlib import Path

import nbformat

from unfallatlas.presentation.assets import (
    AssetStore,
    copy_shared_assets,
    prepare_notebook_assets,
)
from unfallatlas.presentation.models import CellCounts, NotebookAnalysis, NotebookStatus


def _analysis(tmp_path: Path, outputs: list[nbformat.NotebookNode]) -> NotebookAnalysis:
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("chart", execution_count=1, outputs=outputs)]
    )
    return NotebookAnalysis(
        source=tmp_path / "notebooks" / "rich-output.ipynb",
        notebook=notebook,
        title="Rich output",
        status=NotebookStatus.READY,
        counts=CellCounts(0, 1, 0, 0, 0, 1, 0),
        findings=(),
        snapshot_sha256="a" * 64,
        source_sha256="b" * 64,
        output_bytes=0,
    )


def test_asset_store_uses_deterministic_digest_paths_and_deduplicates_bytes(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    data = b"same bytes"

    first = store.put_bytes(
        namespace="images",
        stem="chart",
        suffix=".png",
        data=data,
        media_type="image/png",
        kind="image",
        cell_index=0,
    )
    second = store.put_bytes(
        namespace="images",
        stem="chart",
        suffix=".png",
        data=data,
        media_type="image/png",
        kind="image",
        cell_index=1,
    )

    digest = hashlib.sha256(data).hexdigest()
    assert first.relative_path == Path("assets/images") / f"chart-{digest[:16]}.png"
    assert second.relative_path == first.relative_path
    assert (tmp_path / first.relative_path).read_bytes() == data


def test_plotly_is_externalized_without_mutating_source_notebook(tmp_path: Path) -> None:
    plotly = {"data": [{"type": "bar", "x": ["A"], "y": [3]}], "layout": {}}
    output = nbformat.v4.new_output(
        "display_data",
        data={"application/vnd.plotly.v1+json": plotly, "text/plain": "Figure"},
    )
    analysis = _analysis(tmp_path, [output])
    original = copy.deepcopy(plotly)

    prepared = prepare_notebook_assets(analysis, AssetStore(tmp_path / "site"))

    record = next(asset for asset in prepared.assets if asset.kind == "plotly")
    payload = (tmp_path / "site" / record.relative_path).read_text(encoding="utf-8")
    assert payload.startswith("window.UnfallatlasPresentation.registerPlotlyPayload(")
    assert '"data"' in payload
    assert analysis.notebook.cells[0].outputs[0].data["application/vnd.plotly.v1+json"] == original
    assert prepared.notebook is not analysis.notebook
    metadata = prepared.notebook.cells[0].outputs[0].metadata["unfallatlas_presentation"]
    assert metadata["kind"] == "plotly"
    assert metadata["payload_key"].startswith("plotly-")
    assert metadata["asset_href"] == f"../{record.relative_path.as_posix()}"
    assert metadata["size_bytes"] == record.size_bytes


def test_equal_plotly_bundles_share_asset_but_keep_distinct_chart_ids(tmp_path: Path) -> None:
    bundle = {"data": [{"x": [1], "y": [2]}], "layout": {}}
    outputs = [
        nbformat.v4.new_output(
            "display_data", data={"application/vnd.plotly.v1+json": copy.deepcopy(bundle)}
        ),
        nbformat.v4.new_output(
            "display_data", data={"application/vnd.plotly.v1+json": copy.deepcopy(bundle)}
        ),
    ]
    prepared = prepare_notebook_assets(_analysis(tmp_path, outputs), AssetStore(tmp_path / "site"))

    plotly_assets = [asset for asset in prepared.assets if asset.kind == "plotly"]
    assert len(plotly_assets) == 1
    metadata = [
        output.metadata["unfallatlas_presentation"] for output in prepared.notebook.cells[0].outputs
    ]
    assert metadata[0]["chart_id"] != metadata[1]["chart_id"]
    assert metadata[0]["payload_key"] == metadata[1]["payload_key"]
    assert metadata[0]["asset_href"] == metadata[1]["asset_href"]


def test_images_are_extracted_with_correct_suffix_and_fallbacks_preserved(tmp_path: Path) -> None:
    png = base64.b64encode(b"png-data").decode("ascii")
    jpeg = base64.b64encode(b"jpeg-data").decode("ascii")
    outputs = [
        nbformat.v4.new_output("display_data", data={"image/png": png, "text/plain": "png"}),
        nbformat.v4.new_output("display_data", data={"image/jpeg": jpeg}),
        nbformat.v4.new_output(
            "display_data", data={"image/svg+xml": "<svg></svg>", "text/html": "<b>svg</b>"}
        ),
    ]

    prepared = prepare_notebook_assets(_analysis(tmp_path, outputs), AssetStore(tmp_path / "site"))

    suffixes = {asset.relative_path.suffix for asset in prepared.assets}
    assert suffixes == {".png", ".jpg", ".svg"}
    copied_outputs = prepared.notebook.cells[0].outputs
    assert copied_outputs[0].data["text/plain"] == "png"
    assert copied_outputs[2].data["text/html"] == "<b>svg</b>"
    for output in copied_outputs:
        href = output.metadata["unfallatlas_presentation"]["asset_href"]
        assert href.startswith("../assets/notebooks/rich-output/")


def test_copy_shared_assets_copies_one_plotly_mathjax_css_and_javascript(
    tmp_path: Path, monkeypatch
) -> None:
    import unfallatlas.presentation.assets as assets_module

    static_dir = tmp_path / "package-static"
    static_dir.mkdir()
    (static_dir / "presentation.css").write_bytes(b"body{}")
    (static_dir / "presentation.js").write_bytes(b"window.UnfallatlasPresentation={};")
    monkeypatch.setattr(assets_module, "STATIC_DIR", static_dir)

    records = copy_shared_assets(AssetStore(tmp_path / "site"))

    assert len(records) == 4
    kinds = [record.kind for record in records]
    assert kinds.count("plotly-runtime") == 1
    assert kinds.count("mathjax-runtime") == 1
    assert {record.media_type for record in records} >= {"text/css", "text/javascript"}
    assert all(not record.relative_path.is_absolute() for record in records)
    assert all((tmp_path / "site" / record.relative_path).is_file() for record in records)
    mathjax = next(record for record in records if record.kind == "mathjax-runtime")
    assert mathjax.sha256 == "a4354ff94fd868aea0cc6eaaa79a57fda0588646fc46ee3700a349ee0a11cbe6"
    plotly_records = [record for record in records if record.kind == "plotly-runtime"]
    assert len(plotly_records) == 1
