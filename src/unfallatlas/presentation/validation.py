import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import nbformat
from bs4 import BeautifulSoup
from nbconvert.filters.markdown_mistune import markdown2html_mistune
from nbformat import NotebookNode
from nbformat.validator import ValidationError

from unfallatlas.presentation.discovery import classify_notebook_status, extract_notebook_title
from unfallatlas.presentation.metadata import snapshot_sha256, source_sha256
from unfallatlas.presentation.models import (
    CellCounts,
    Finding,
    NotebookAnalysis,
    NotebookStatus,
    Severity,
)

LARGE_OUTPUT_BYTES = 5 * 1024 * 1024
VERY_LARGE_NOTEBOOK_OUTPUT_BYTES = 100 * 1024 * 1024

_PLOTLY_MIME = "application/vnd.plotly.v1+json"
WIDGET_VIEW_MIME = "application/vnd.jupyter.widget-view+json"
_WIDGET_STATE_MIME = "application/vnd.jupyter.widget-state+json"
_DATA_WRANGLER_MIMES = {
    "application/vnd.dataresource+json",
    "application/vnd.data-wrangler+json",
}
_SUPPORTED_MIME_PRIORITY = (
    _PLOTLY_MIME,
    "text/html",
    "image/svg+xml",
    "image/png",
    "image/jpeg",
    "text/latex",
    "text/plain",
    "application/javascript",
    "text/javascript",
)
_WIDGET_STATIC_FALLBACK_PRIORITY = (
    "text/html",
    "image/svg+xml",
    "image/png",
    "image/jpeg",
    "text/latex",
    "text/plain",
)
_ACTIVE_HTML_TAGS = ("script", "iframe", "object", "embed", "canvas")
_RESOURCE_ATTRIBUTES = (
    ("img", "src"),
    ("script", "src"),
    ("iframe", "src"),
    ("source", "src"),
    ("video", "src"),
    ("audio", "src"),
)
_CSS_URL = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", flags=re.IGNORECASE)
_MARKDOWN_TABLE_DELIMITER = re.compile(r"(?m)^\s*\|?\s*:?-{3,}.*\|\s*$")
_EXTERNAL_MAP_STYLES = {
    "basic",
    "carto-darkmatter",
    "carto-positron",
    "dark",
    "light",
    "open-street-map",
    "outdoors",
    "satellite",
    "satellite-streets",
    "stamen-terrain",
    "stamen-toner",
    "stamen-watercolor",
    "streets",
}


class NotebookReadError(RuntimeError):
    """Raised when a notebook cannot be decoded or fails nbformat validation."""


def _finding(
    code: str,
    severity: Severity,
    message: str,
    cell_index: int | None = None,
    *,
    strict: bool = False,
) -> Finding:
    return Finding(
        code=code,
        severity=severity,
        message=message,
        cell_index=cell_index,
        strict_blocker=strict,
    )


def validate_rendered_html(html: str) -> tuple[Finding, ...]:
    """Report Markdown table syntax that leaked into rendered content blocks."""
    soup = BeautifulSoup(html, "html.parser")
    findings: list[Finding] = []
    for node in soup.select("p, li"):
        text = node.get_text("\n", strip=True)
        if "|" in text and _MARKDOWN_TABLE_DELIMITER.search(text):
            findings.append(
                _finding(
                    "literal-markdown-table",
                    Severity.ERROR,
                    "Markdown table syntax was emitted as paragraph text.",
                )
            )
    return tuple(findings)


def _resource_finding(
    reference: str,
    notebook_dir: Path,
    repo_root: Path,
    cell_index: int,
) -> Finding | None:
    value = reference.strip()
    if not value or value.startswith("#"):
        return None

    parsed = urlsplit(value)
    scheme = parsed.scheme.casefold()
    if scheme in {"data", "mailto"}:
        return None
    if value.startswith("//") or scheme in {"http", "https"}:
        return _finding(
            "EXTERNAL_RUNTIME_RESOURCE",
            Severity.ERROR,
            f"Runtime resource requires network access: {value}",
            cell_index,
            strict=True,
        )
    if scheme:
        return _finding(
            "EXTERNAL_RUNTIME_RESOURCE",
            Severity.ERROR,
            f"Runtime resource uses a non-local URI: {value}",
            cell_index,
            strict=True,
        )

    local_path = unquote(parsed.path)
    if not local_path:
        return None
    candidate = (notebook_dir / local_path).resolve(strict=False)
    root = repo_root.resolve(strict=False)
    if not candidate.is_relative_to(root):
        return _finding(
            "UNSAFE_LOCAL_ASSET",
            Severity.ERROR,
            f"Local asset escapes the repository: {value}",
            cell_index,
            strict=True,
        )
    if not candidate.is_file():
        return _finding(
            "MISSING_LOCAL_ASSET",
            Severity.ERROR,
            f"Local asset does not exist: {value}",
            cell_index,
            strict=True,
        )
    return None


def scan_runtime_references(
    html: str,
    notebook_dir: Path,
    repo_root: Path,
    cell_index: int,
) -> tuple[Finding, ...]:
    """Find network-dependent, missing, and repository-escaping HTML resources."""
    soup = BeautifulSoup(html, "html.parser")
    references: list[str] = []
    for tag_name, attribute in _RESOURCE_ATTRIBUTES:
        references.extend(
            str(tag[attribute]) for tag in soup.find_all(tag_name) if tag.get(attribute)
        )
    references.extend(
        str(tag["href"])
        for tag in soup.find_all("link")
        if tag.get("href") and "stylesheet" in {str(item).casefold() for item in tag.get("rel", [])}
    )
    css_chunks = [tag.get_text() for tag in soup.find_all("style")]
    css_chunks.extend(str(tag["style"]) for tag in soup.find_all(style=True))
    for css in css_chunks:
        references.extend(match.group(2) for match in _CSS_URL.finditer(css))

    findings = [
        finding
        for reference in references
        if (finding := _resource_finding(reference, notebook_dir, repo_root, cell_index))
        is not None
    ]
    return tuple(findings)


def _compact_json_bytes(value: Any) -> int:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return len(payload.encode("utf-8"))


def _widget_state(nb: NotebookNode) -> dict[str, Any]:
    widgets = nb.metadata.get("widgets", {})
    state_bundle = widgets.get(_WIDGET_STATE_MIME, {}) if isinstance(widgets, dict) else {}
    state = state_bundle.get("state", {}) if isinstance(state_bundle, dict) else {}
    return state if isinstance(state, dict) else {}


def _selected_mime(data: dict[str, Any]) -> str | None:
    for mime in _SUPPORTED_MIME_PRIORITY:
        if mime in data:
            return mime
    return None


def _static_html_is_visible(value: Any) -> bool:
    if isinstance(value, list):
        value = "".join(str(item) for item in value)
    if not isinstance(value, str) or not value.strip():
        return False

    soup = BeautifulSoup(value, "html.parser")
    if soup.find(_ACTIVE_HTML_TAGS) is not None:
        return False
    for tag in soup.find_all(True):
        for attribute, raw_value in tag.attrs.items():
            if attribute.casefold().startswith("on"):
                return False
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            if any("javascript:" in str(item).casefold() for item in values):
                return False

    for hidden in soup.find_all(("style", "template", "noscript")):
        hidden.decompose()
    if soup.get_text(" ", strip=True):
        return True
    if any(tag.get("src") for tag in soup.find_all(("img", "video", "audio"))):
        return True
    if soup.find("hr") is not None:
        return True
    return any(svg.contents for svg in soup.find_all("svg"))


def select_widget_static_fallback(data: dict[str, Any]) -> str | None:
    for mime in _WIDGET_STATIC_FALLBACK_PRIORITY:
        if mime not in data:
            continue
        if mime == "text/html" and not _static_html_is_visible(data[mime]):
            continue
        return mime
    return None


def _contains_remote_reference(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("//") or urlsplit(value).scheme in {"http", "https", "mapbox"}
    if isinstance(value, dict):
        return any(_contains_remote_reference(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_remote_reference(item) for item in value)
    return False


def _map_uses_external_tiles(layout: dict[str, Any]) -> bool:
    for container_name, container in layout.items():
        if not re.fullmatch(r"(?:mapbox|map)\d*", container_name):
            continue
        if not isinstance(container, dict):
            continue
        style = container.get("style")
        if isinstance(style, str):
            normalized = style.casefold()
            if normalized in _EXTERNAL_MAP_STYLES or _contains_remote_reference(style):
                return True
        elif _contains_remote_reference(style):
            return True
        if _contains_remote_reference(container.get("layers")):
            return True
        if container.get("layers"):
            return True
    return False


def _plotly_findings(
    payload: Any,
    notebook_dir: Path,
    repo_root: Path,
    cell_index: int,
) -> tuple[Finding, ...]:
    if not isinstance(payload, dict):
        return ()
    layout = payload.get("layout", {})
    if not isinstance(layout, dict):
        return ()

    findings: list[Finding] = []
    if _map_uses_external_tiles(layout):
        findings.append(
            _finding(
                "EXTERNAL_MAP_TILES",
                Severity.ERROR,
                "Plotly map requires external tile resources.",
                cell_index,
                strict=True,
            )
        )
    for image in layout.get("images", []):
        if not isinstance(image, dict) or not isinstance(image.get("source"), str):
            continue
        finding = _resource_finding(image["source"], notebook_dir, repo_root, cell_index)
        if finding is not None:
            findings.append(finding)
    return tuple(findings)


def _output_findings(
    output: NotebookNode,
    nb: NotebookNode,
    notebook_dir: Path,
    repo_root: Path,
    cell_index: int,
) -> tuple[Finding, ...]:
    data = output.get("data")
    if not isinstance(data, dict):
        return ()

    has_widget = WIDGET_VIEW_MIME in data
    selected = select_widget_static_fallback(data) if has_widget else _selected_mime(data)
    findings: list[Finding] = []
    if has_widget:
        has_fallback = selected is not None
        findings.append(
            _finding(
                "WIDGET_UNSUPPORTED",
                Severity.WARNING if has_fallback else Severity.ERROR,
                (
                    "Interactive widget output is unsupported; using a stored static fallback."
                    if has_fallback
                    else "Interactive widget output is unsupported and has no static fallback."
                ),
                cell_index,
                strict=not has_fallback,
            )
        )
        view = data[WIDGET_VIEW_MIME]
        model_id = view.get("model_id") if isinstance(view, dict) else None
        if (not model_id or model_id not in _widget_state(nb)) and not has_fallback:
            findings.append(
                _finding(
                    "WIDGET_STATE_MISSING",
                    Severity.ERROR,
                    "Widget output has no matching embedded notebook state or static fallback.",
                    cell_index,
                    strict=True,
                )
            )

    if selected is None:
        findings.append(
            _finding(
                "UNSUPPORTED_MIME",
                Severity.ERROR,
                f"Output has no supported MIME representation: {', '.join(sorted(data)) or 'none'}",
                cell_index,
                strict=True,
            )
        )
        return tuple(findings)

    if selected == "text/html" and isinstance(data[selected], str):
        findings.extend(
            scan_runtime_references(data[selected], notebook_dir, repo_root, cell_index)
        )
    elif selected == _PLOTLY_MIME:
        findings.extend(_plotly_findings(data[selected], notebook_dir, repo_root, cell_index))

    if set(data) & _DATA_WRANGLER_MIMES and not ({"text/html", "text/plain"} & set(data)):
        findings.append(
            _finding(
                "UNSUPPORTED_MIME",
                Severity.ERROR,
                "Data Wrangler output requires an HTML or plain-text fallback.",
                cell_index,
                strict=True,
            )
        )
    return tuple(findings)


def _read_notebook(source: Path) -> NotebookNode:
    try:
        with source.open(encoding="utf-8") as handle:
            notebook = nbformat.read(handle, as_version=4)
        nbformat.validate(notebook)
    except (OSError, UnicodeError, ValueError, ValidationError) as error:
        raise NotebookReadError(f"Could not read valid notebook {source}: {error}") from error
    return notebook


def read_and_validate_notebook(source: Path, repo_root: Path) -> NotebookAnalysis:
    """Read a notebook without executing it and validate its stored presentation snapshot."""
    notebook = _read_notebook(source)
    snapshot_digest = snapshot_sha256(notebook)
    source_digest = source_sha256(notebook)
    status = classify_notebook_status(notebook)
    findings: list[Finding] = []
    markdown_count = 0
    code_count = 0
    raw_count = 0
    unexecuted_count = 0
    executed_without_output_count = 0
    code_with_output_count = 0
    error_output_count = 0
    output_bytes = 0
    previous_execution_count: int | None = None
    seen_execution_counts: set[int] = set()

    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown":
            markdown_count += 1
            if cell.source:
                rendered = markdown2html_mistune(cell.source)
                findings.extend(
                    scan_runtime_references(rendered, source.parent, repo_root, cell_index)
                )
            continue
        if cell.cell_type == "raw":
            raw_count += 1
            continue
        if cell.cell_type != "code" or not cell.source.strip():
            continue

        code_count += 1
        execution_count = cell.get("execution_count")
        outputs = cell.get("outputs", [])
        if execution_count is None:
            unexecuted_count += 1
            findings.append(
                _finding(
                    "UNEXECUTED_CELL",
                    Severity.WARNING,
                    "Non-empty code cell has not been executed.",
                    cell_index,
                    strict=True,
                )
            )
        else:
            if execution_count in seen_execution_counts:
                findings.append(
                    _finding(
                        "DUPLICATE_EXECUTION_COUNT",
                        Severity.WARNING,
                        f"Execution count {execution_count} is duplicated.",
                        cell_index,
                        strict=True,
                    )
                )
            if previous_execution_count is not None and execution_count < previous_execution_count:
                findings.append(
                    _finding(
                        "NON_MONOTONIC_EXECUTION",
                        Severity.WARNING,
                        "Execution counts decrease in notebook order.",
                        cell_index,
                        strict=True,
                    )
                )
            seen_execution_counts.add(execution_count)
            previous_execution_count = execution_count

        if outputs:
            code_with_output_count += 1
        elif execution_count is not None:
            executed_without_output_count += 1
            findings.append(
                _finding(
                    "EXECUTED_NO_OUTPUT",
                    Severity.INFO,
                    "Code cell was executed and legitimately stored no output.",
                    cell_index,
                )
            )

        for output in outputs:
            payload_bytes = _compact_json_bytes(output)
            output_bytes += payload_bytes
            if payload_bytes > LARGE_OUTPUT_BYTES:
                findings.append(
                    _finding(
                        "LARGE_OUTPUT",
                        Severity.WARNING,
                        f"Stored output is {payload_bytes} bytes.",
                        cell_index,
                    )
                )
            if output.get("output_type") == "error":
                error_output_count += 1
                findings.append(
                    _finding(
                        "ERROR_OUTPUT",
                        Severity.ERROR,
                        "Code cell contains a stored error output.",
                        cell_index,
                        strict=True,
                    )
                )
            findings.extend(
                _output_findings(output, notebook, source.parent, repo_root, cell_index)
            )

    if output_bytes > VERY_LARGE_NOTEBOOK_OUTPUT_BYTES:
        findings.append(
            _finding(
                "VERY_LARGE_NOTEBOOK_OUTPUT",
                Severity.WARNING,
                f"Notebook stores {output_bytes} output bytes in total.",
            )
        )
    if status is NotebookStatus.PLACEHOLDER:
        findings.append(
            _finding(
                "PLACEHOLDER_NOTEBOOK",
                Severity.WARNING,
                "Notebook is classified as a placeholder.",
            )
        )
    elif status is NotebookStatus.WIP:
        findings.append(
            _finding(
                "WIP_NOTEBOOK",
                Severity.WARNING,
                "Notebook is classified as work in progress.",
                strict=True,
            )
        )

    return NotebookAnalysis(
        source=source,
        notebook=notebook,
        title=extract_notebook_title(notebook, source.stem),
        status=status,
        counts=CellCounts(
            markdown=markdown_count,
            code=code_count,
            raw=raw_count,
            unexecuted_code=unexecuted_count,
            executed_without_output=executed_without_output_count,
            code_with_output=code_with_output_count,
            error_outputs=error_output_count,
        ),
        findings=tuple(findings),
        snapshot_sha256=snapshot_digest,
        source_sha256=source_digest,
        output_bytes=output_bytes,
    )
