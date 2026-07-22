from __future__ import annotations

import mimetypes
import os
import warnings
from dataclasses import asdict, replace
from pathlib import Path, PurePath, PureWindowsPath
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from bs4 import BeautifulSoup
from nbconvert import HTMLExporter
from nbconvert.preprocessors.csshtmlheader import CSSHTMLHeaderPreprocessor
from traitlets.config import Config

from unfallatlas.presentation.assets import (
    AssetStore,
    asset_href,
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
        # Highlight2HTML (the filter that actually tokenizes code cell
        # source into `.highlight .k`/`.s`/... spans) reads
        # `extra_formatter_options` from traitlets config on its parent
        # (this exporter) - passed straight through to Pygments'
        # HtmlFormatter. Set it before super().__init__() so the filter
        # picks it up the first time it is instantiated in
        # HTMLExporter.from_notebook_node(). linenos="table" renders the
        # standard two-column line-number gutter + code layout used by
        # GitHub/VS Code, not inline numbers mixed into the copyable text.
        config = kwargs.pop("config", None) or Config()
        config.Highlight2HTML.extra_formatter_options = {"linenos": "table"}
        kwargs["config"] = config
        kwargs.setdefault("template_name", "notebook")
        kwargs.setdefault("extra_template_basedirs", [str(PACKAGE_TEMPLATES_ROOT)])
        super().__init__(**kwargs)
        self.register_filter("classify_html_output", classify_html_output)
        self.register_filter("nest_toc", nest_toc)
        # The "notebook" template extends nbconvert's bare "basic" template
        # (no built-in <style> block), and HTMLExporter does not enable
        # CSSHTMLHeaderPreprocessor by default outside the "lab"/"classic"
        # templates. Without it, resources["inlining"]["css"] - which
        # templates/notebook/index.html.j2 loops over - stays empty, so
        # code cells get Pygments' `.highlight .k`/`.s`/... span classes
        # with no matching CSS rules and render as unstyled black text.
        # "one-dark" mirrors the Atom/VS Code "One Dark Pro" theme
        # (#282c34 slate background, purple keywords, cyan operators,
        # muted-gray comments) - a widely recognized, professional look;
        # the surrounding page chrome stays light (see presentation.css),
        # so only the code panels themselves are dark, like a docs site
        # embedding a syntax-highlighted snippet.
        self.register_preprocessor(
            CSSHTMLHeaderPreprocessor(style="one-dark"),
            enabled=True,
        )


def _href(record: AssetRecord, href_prefix: Path = Path("..")) -> str:
    return asset_href(record.relative_path, href_prefix)


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


def _safe_output_relative_path(path: PurePath) -> Path:
    native_path = Path(path)
    windows_path = PureWindowsPath(str(path))
    if (
        native_path.is_absolute()
        or native_path == Path(".")
        or ".." in native_path.parts
        or windows_path.drive
        or windows_path.root
        or windows_path == PureWindowsPath(".")
        or ".." in windows_path.parts
    ):
        raise ValueError(f"output_relative_path must be a safe relative path: {path}")
    return Path(*windows_path.parts)


def _asset_href_prefix(destination: Path, output_root: Path) -> Path:
    relative_parent = destination.parent.relative_to(output_root)
    return Path(*(len(relative_parent.parts) * ("..",)))


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
    published_parts = Path(unquote(parsed.path)).parts
    while published_parts and published_parts[0] == "..":
        published_parts = published_parts[1:]
    if published_parts and published_parts[0] == "assets":
        published_root = (output_root / "assets").resolve(strict=False)
        published = (output_root / Path(*published_parts)).resolve(strict=False)
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
    rewrites: dict[str, str],
    allow_writes: bool,
    href_prefix: Path,
) -> str:
    if reference in rewrites:
        return rewrites[reference]
    source = _local_source(reference, analysis.source, repo_root, output_root)
    if source is None:
        return reference
    if not allow_writes:
        raise RuntimeError(f"final render introduced an undiscovered local resource: {reference}")
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
    rewritten = urlunsplit(("", "", _href(record, href_prefix), parsed.query, parsed.fragment))
    rewrites[reference] = rewritten
    return rewritten


def _publish_local_resources(
    html: str,
    *,
    analysis: NotebookAnalysis,
    repo_root: Path | None,
    output_root: Path,
    store: AssetStore,
    known_rewrites: dict[str, str] | None = None,
    allow_writes: bool = True,
    href_prefix: Path = Path(".."),
) -> tuple[str, tuple[AssetRecord, ...], dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    for base in soup.find_all("base"):
        base.decompose()
    records: dict[Path, AssetRecord] = {}
    rewrites = known_rewrites if known_rewrites is not None else {}

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
                    rewrites=rewrites,
                    allow_writes=allow_writes,
                    href_prefix=href_prefix,
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
                rewrites=rewrites,
                allow_writes=allow_writes,
                href_prefix=href_prefix,
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
                rewrites=rewrites,
                allow_writes=allow_writes,
                href_prefix=href_prefix,
            )
            return f"url({quote}{reference}{quote})"

        return _CSS_URL.sub(replacement, css)

    for tag in soup.find_all("style"):
        tag.string = rewrite_css(tag.get_text())
    for tag in soup.find_all(style=True):
        tag["style"] = rewrite_css(str(tag["style"]))
    for iframe in soup.find_all("iframe", srcdoc=True):
        rewritten_srcdoc, nested_records, _ = _publish_local_resources(
            str(iframe["srcdoc"]),
            analysis=analysis,
            repo_root=repo_root,
            output_root=output_root,
            store=store,
            known_rewrites=rewrites,
            allow_writes=allow_writes,
            href_prefix=href_prefix,
        )
        iframe["srcdoc"] = rewritten_srcdoc
        for record in nested_records:
            records.setdefault(record.relative_path, record)
    return str(soup), tuple(records.values()), rewrites


def _presentation_resources(
    analysis: NotebookAnalysis,
    metadata: ExportMetadata,
    notebook_assets: tuple[AssetRecord, ...],
    shared_assets: tuple[AssetRecord, ...],
    href_prefix: Path = Path(".."),
    repo_root: Path | None = None,
) -> dict[str, object]:
    all_assets = (*shared_assets, *notebook_assets)
    source = analysis.source.resolve(strict=False)
    try:
        root = _repo_root_for(analysis.source, repo_root)
    except ValueError:
        source_path = source.name
    else:
        source_path = (
            source.relative_to(root).as_posix() if source.is_relative_to(root) else source.name
        )
    warning_count = sum(finding.severity.value == "warning" for finding in analysis.findings)
    execution_complete = (
        analysis.status.value == "ready"
        and analysis.counts.unexecuted_code == 0
        and analysis.counts.error_outputs == 0
    )
    return {
        "title": analysis.title,
        "status": analysis.status.value,
        "findings": tuple(asdict(finding) for finding in analysis.findings),
        "counts": asdict(analysis.counts),
        "toc": build_toc(analysis.notebook),
        "metadata": asdict(metadata),
        "snapshot_sha256": analysis.snapshot_sha256,
        "source_path": source_path,
        "warning_count": warning_count,
        "execution_complete": execution_complete,
        "index_href": asset_href(Path("index.html"), href_prefix),
        "style_href": _href(_record_by_kind(shared_assets, "ui-style"), href_prefix),
        "script_href": _href(_record_by_kind(shared_assets, "ui-script"), href_prefix),
        "plotly_runtime_href": _href(_record_by_kind(shared_assets, "plotly-runtime"), href_prefix),
        "mathjax_runtime_href": _href(
            _record_by_kind(shared_assets, "mathjax-runtime"), href_prefix
        ),
        "asset_map": _asset_map(all_assets),
    }


def render_notebook(
    analysis: NotebookAnalysis,
    metadata: ExportMetadata,
    output_root: Path,
    *,
    repo_root: Path | None = None,
    output_relative_path: Path | None = None,
) -> ExportResult:
    """Render saved notebook state to one atomically published offline HTML document."""
    output_root = Path(output_root)
    destination = output_root / "notebooks" / f"{analysis.source.stem}.html"
    assets: list[AssetRecord] = []

    try:
        if output_relative_path is not None:
            destination = (
                output_root / "notebooks" / _safe_output_relative_path(output_relative_path)
            )
        href_prefix = _asset_href_prefix(destination, output_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        store = AssetStore(output_root)
        shared_assets = copy_shared_assets(store)
        assets.extend(shared_assets)

        anchored_notebook = add_stable_heading_anchors(analysis.notebook)
        anchored_analysis = replace(analysis, notebook=anchored_notebook)
        prepared = prepare_notebook_assets(
            anchored_analysis,
            store,
            href_prefix=href_prefix,
            repo_root=repo_root,
        )
        assets.extend(prepared.assets)

        exporter = PresentationHTMLExporter(
            template_name="notebook",
            extra_template_basedirs=[str(PACKAGE_TEMPLATES_ROOT)],
        )
        exporter.exclude_input_prompt = True
        exporter.exclude_output_prompt = True
        exporter.embed_images = False
        draft_presentation = _presentation_resources(
            anchored_analysis,
            metadata,
            prepared.assets,
            shared_assets,
            href_prefix,
            repo_root,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*application/vnd\.plotly\.v1\+json.*",
                category=UserWarning,
            )
            draft_html, _ = exporter.from_notebook_node(
                prepared.notebook,
                resources={"presentation": draft_presentation},
            )
            _, local_assets, rewrites = _publish_local_resources(
                draft_html,
                analysis=analysis,
                repo_root=repo_root,
                output_root=output_root,
                store=store,
                href_prefix=href_prefix,
            )
            assets.extend(local_assets)
            presentation = _presentation_resources(
                anchored_analysis,
                metadata,
                (*prepared.assets, *local_assets),
                shared_assets,
                href_prefix,
                repo_root,
            )
            html, _ = exporter.from_notebook_node(
                prepared.notebook,
                resources={"presentation": presentation},
            )
            html, final_assets, _ = _publish_local_resources(
                html,
                analysis=analysis,
                repo_root=repo_root,
                output_root=output_root,
                store=store,
                known_rewrites=rewrites,
                allow_writes=False,
                href_prefix=href_prefix,
            )
            if final_assets:
                raise RuntimeError("final render unexpectedly republished local resources")
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
