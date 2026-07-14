from __future__ import annotations

import os
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from nbconvert import HTMLExporter

from unfallatlas.presentation.assets import (
    AssetStore,
    copy_shared_assets,
    prepare_notebook_assets,
)
from unfallatlas.presentation.models import (
    AssetRecord,
    ExportMetadata,
    ExportResult,
    NotebookAnalysis,
    add_stable_heading_anchors,
    build_toc,
    classify_html_output,
    nest_toc,
)

PACKAGE_TEMPLATES_ROOT = Path(__file__).parent / "templates"


class PresentationHTMLExporter(HTMLExporter):
    """nbconvert HTML exporter configured with presentation template helpers."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("template_name", "notebook")
        kwargs.setdefault("extra_template_basedirs", [str(PACKAGE_TEMPLATES_ROOT)])
        super().__init__(**kwargs)
        self.register_filter("classify_html_output", classify_html_output)
        self.register_filter("nest_toc", nest_toc)


def _href(record: AssetRecord) -> str:
    return f"../{record.relative_path.as_posix()}"


def _record_by_kind(records: tuple[AssetRecord, ...], kind: str) -> AssetRecord:
    return next(record for record in records if record.kind == kind)


def _write_html_atomic(target: Path, data: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _asset_map(records: tuple[AssetRecord, ...]) -> dict[str, dict[str, object]]:
    return {
        record.relative_path.as_posix(): {
            "sha256": record.sha256,
            "size_bytes": record.size_bytes,
            "media_type": record.media_type,
            "kind": record.kind,
            "cell_index": record.cell_index,
        }
        for record in records
    }


def _presentation_resources(
    analysis: NotebookAnalysis,
    metadata: ExportMetadata,
    notebook_assets: tuple[AssetRecord, ...],
    shared_assets: tuple[AssetRecord, ...],
) -> dict[str, object]:
    all_assets = (*shared_assets, *notebook_assets)
    return {
        "title": analysis.title,
        "status": analysis.status.value,
        "findings": tuple(asdict(finding) for finding in analysis.findings),
        "counts": asdict(analysis.counts),
        "toc": build_toc(analysis.notebook),
        "metadata": asdict(metadata),
        "snapshot_sha256": analysis.snapshot_sha256,
        "source_path": analysis.source.as_posix(),
        "style_href": _href(_record_by_kind(shared_assets, "ui-style")),
        "script_href": _href(_record_by_kind(shared_assets, "ui-script")),
        "plotly_runtime_href": _href(_record_by_kind(shared_assets, "plotly-runtime")),
        "mathjax_runtime_href": _href(_record_by_kind(shared_assets, "mathjax-runtime")),
        "asset_map": _asset_map(all_assets),
    }


def render_notebook(
    analysis: NotebookAnalysis,
    metadata: ExportMetadata,
    output_root: Path,
) -> ExportResult:
    """Render saved notebook state to one atomically published offline HTML document."""
    output_root = Path(output_root)
    destination = output_root / "notebooks" / f"{analysis.source.stem}.html"
    assets: tuple[AssetRecord, ...] = ()

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        store = AssetStore(output_root)
        shared_assets = copy_shared_assets(store)

        anchored_notebook = add_stable_heading_anchors(analysis.notebook)
        anchored_analysis = replace(analysis, notebook=anchored_notebook)
        prepared = prepare_notebook_assets(anchored_analysis, store)
        assets = (*shared_assets, *prepared.assets)

        exporter = PresentationHTMLExporter(
            template_name="notebook",
            extra_template_basedirs=[str(PACKAGE_TEMPLATES_ROOT)],
        )
        exporter.exclude_input_prompt = True
        exporter.exclude_output_prompt = True
        exporter.embed_images = False
        presentation = _presentation_resources(
            anchored_analysis,
            metadata,
            prepared.assets,
            shared_assets,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*application/vnd\.plotly\.v1\+json.*",
                category=UserWarning,
            )
            html, _ = exporter.from_notebook_node(
                prepared.notebook,
                resources={"presentation": presentation},
            )
        rendered = html.encode("utf-8")
        _write_html_atomic(destination, rendered)
    except Exception as exc:
        return ExportResult(
            source=analysis.source,
            destination=destination,
            status=analysis.status,
            findings=analysis.findings,
            size_bytes=0,
            error=str(exc) or type(exc).__name__,
            assets=assets,
        )

    return ExportResult(
        source=analysis.source,
        destination=destination,
        status=analysis.status,
        findings=analysis.findings,
        size_bytes=len(rendered),
        error=None,
        assets=assets,
    )
