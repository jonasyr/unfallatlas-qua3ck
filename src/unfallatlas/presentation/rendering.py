from __future__ import annotations

import mimetypes
import os
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from bs4 import BeautifulSoup
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
from unfallatlas.presentation.validation import _CSS_URL, _RESOURCE_ATTRIBUTES

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


def _repo_root_for(source: Path, explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return Path(explicit_root).resolve(strict=False)
    source_path = source.resolve(strict=False)
    for candidate in (source_path.parent, *source_path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / ".git").exists():
            return candidate
    raise ValueError(f"cannot determine repository root for local resources in {source}")


def _local_source(
    reference: str,
    source: Path,
    repo_root: Path | None,
    output_root: Path,
) -> Path | None:
    value = reference.strip()
    if not value or value.startswith("#"):
        return None
    parsed = urlsplit(value)
    if value.startswith("//") or parsed.scheme or not parsed.path:
        return None
    if parsed.path == "../assets" or parsed.path.startswith("../assets/"):
        published_root = (output_root / "assets").resolve(strict=False)
        published = (output_root / "notebooks" / unquote(parsed.path)).resolve(strict=False)
        if not published.is_relative_to(published_root):
            raise ValueError(f"published resource escapes output assets: {value}")
        if not published.is_file():
            raise FileNotFoundError(f"published resource does not exist: {value}")
        return None

    root = _repo_root_for(source, repo_root)
    candidate = (source.parent / unquote(parsed.path)).resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise ValueError(f"local resource escapes repository: {value}")
    if not candidate.is_file():
        raise FileNotFoundError(f"local resource does not exist: {value}")
    return candidate


def _publish_reference(
    reference: str,
    *,
    analysis: NotebookAnalysis,
    repo_root: Path | None,
    output_root: Path,
    store: AssetStore,
    records: dict[Path, AssetRecord],
) -> str:
    source = _local_source(reference, analysis.source, repo_root, output_root)
    if source is None:
        return reference
    media_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
    record = store.put_bytes(
        namespace=f"notebooks/{analysis.source.stem}/local",
        stem=f"resource-{source.stem}",
        suffix=source.suffix,
        data=source.read_bytes(),
        media_type=media_type,
        kind="local-resource",
        cell_index=None,
    )
    records.setdefault(record.relative_path, record)
    parsed = urlsplit(reference)
    return urlunsplit(("", "", _href(record), parsed.query, parsed.fragment))


def _publish_local_resources(
    html: str,
    *,
    analysis: NotebookAnalysis,
    repo_root: Path | None,
    output_root: Path,
    store: AssetStore,
) -> tuple[str, tuple[AssetRecord, ...]]:
    soup = BeautifulSoup(html, "html.parser")
    records: dict[Path, AssetRecord] = {}

    for tag_name, attribute in _RESOURCE_ATTRIBUTES:
        for tag in soup.find_all(tag_name):
            if tag.get(attribute):
                tag[attribute] = _publish_reference(
                    str(tag[attribute]),
                    analysis=analysis,
                    repo_root=repo_root,
                    output_root=output_root,
                    store=store,
                    records=records,
                )
    for tag in soup.find_all("link"):
        relations = {str(item).casefold() for item in tag.get("rel", [])}
        if tag.get("href") and "stylesheet" in relations:
            tag["href"] = _publish_reference(
                str(tag["href"]),
                analysis=analysis,
                repo_root=repo_root,
                output_root=output_root,
                store=store,
                records=records,
            )

    def rewrite_css(css: str) -> str:
        def replacement(match: Any) -> str:
            quote = match.group(1)
            reference = _publish_reference(
                match.group(2),
                analysis=analysis,
                repo_root=repo_root,
                output_root=output_root,
                store=store,
                records=records,
            )
            return f"url({quote}{reference}{quote})"

        return _CSS_URL.sub(replacement, css)

    for tag in soup.find_all("style"):
        tag.string = rewrite_css(tag.get_text())
    for tag in soup.find_all(style=True):
        tag["style"] = rewrite_css(str(tag["style"]))
    return str(soup), tuple(records.values())


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
    *,
    repo_root: Path | None = None,
) -> ExportResult:
    """Render saved notebook state to one atomically published offline HTML document."""
    output_root = Path(output_root)
    destination = output_root / "notebooks" / f"{analysis.source.stem}.html"
    assets: list[AssetRecord] = []

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        store = AssetStore(output_root)
        shared_assets = copy_shared_assets(store)
        assets.extend(shared_assets)

        anchored_notebook = add_stable_heading_anchors(analysis.notebook)
        anchored_analysis = replace(analysis, notebook=anchored_notebook)
        prepared = prepare_notebook_assets(anchored_analysis, store)
        assets.extend(prepared.assets)

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
        html, local_assets = _publish_local_resources(
            html,
            analysis=analysis,
            repo_root=repo_root,
            output_root=output_root,
            store=store,
        )
        assets.extend(local_assets)
        presentation["asset_map"] = _asset_map(tuple(assets))
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
            assets=tuple(assets),
        )

    return ExportResult(
        source=analysis.source,
        destination=destination,
        status=analysis.status,
        findings=analysis.findings,
        size_bytes=len(rendered),
        error=None,
        assets=tuple(assets),
    )
