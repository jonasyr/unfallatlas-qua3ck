import copy
import html
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from nbformat import NotebookNode

_ATX_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
_ANCHOR_SPAN = re.compile(
    r'^<span id="[^"]+" class="heading-anchor" aria-hidden="true"></span>\n',
    re.MULTILINE,
)


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class NotebookStatus(StrEnum):
    READY = "ready"
    WIP = "wip"
    PLACEHOLDER = "placeholder"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class Finding:
    code: str
    severity: Severity
    message: str
    cell_index: int | None = None
    strict_blocker: bool = False


@dataclass(frozen=True, slots=True)
class CellCounts:
    markdown: int
    code: int
    raw: int
    unexecuted_code: int
    executed_without_output: int
    code_with_output: int
    error_outputs: int


@dataclass(frozen=True, slots=True)
class NotebookAnalysis:
    source: Path
    notebook: NotebookNode
    title: str
    status: NotebookStatus
    counts: CellCounts
    findings: tuple[Finding, ...]
    snapshot_sha256: str
    source_sha256: str
    output_bytes: int

    @property
    def strict_blocked(self) -> bool:
        return any(finding.strict_blocker for finding in self.findings)


@dataclass(frozen=True, slots=True)
class GitMetadata:
    commit: str
    short_commit: str
    branch: str
    dirty: bool


@dataclass(frozen=True, slots=True)
class ExportMetadata:
    exported_at: datetime
    exported_at_local: str
    git: GitMetadata


@dataclass(frozen=True, slots=True)
class TocEntry:
    level: int
    title: str
    anchor: str


@dataclass(frozen=True, slots=True)
class AssetRecord:
    relative_path: Path
    sha256: str
    size_bytes: int
    media_type: str
    kind: str
    cell_index: int | None


@dataclass(frozen=True, slots=True)
class ExportResult:
    source: Path
    destination: Path
    status: NotebookStatus
    findings: tuple[Finding, ...]
    size_bytes: int
    error: str | None
    assets: tuple[AssetRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class BatchResult:
    results: tuple[ExportResult, ...]

    @property
    def successful(self) -> bool:
        return all(result.error is None for result in self.results)


def _plain_heading_title(markdown: str) -> str:
    title = re.sub(r"!??\[([^]]*)\]\([^)]*\)", r"\1", markdown)
    title = re.sub(r"<[^>]+>", "", title)
    title = re.sub(r"[`*_~]", "", title)
    return html.unescape(title).strip()


def _slug(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title).casefold()
    normalized = normalized.translate(str.maketrans({"ä": "a", "ö": "o", "ü": "u", "ß": "ss"}))
    normalized = re.sub(r"\s+", "-", normalized)
    return (
        "".join(
            character
            for character in normalized
            if character in {"_", "-"} or unicodedata.category(character)[0] in {"L", "N"}
        ).strip("-")
        or "abschnitt"
    )


def _toc_entries(notebook: NotebookNode) -> tuple[TocEntry, ...]:
    seen: dict[str, int] = {}
    entries: list[TocEntry] = []
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        source = _ANCHOR_SPAN.sub("", str(cell.source))
        for match in _ATX_HEADING.finditer(source):
            title = _plain_heading_title(match.group(2))
            base = _slug(title)
            seen[base] = seen.get(base, 0) + 1
            suffix = "" if seen[base] == 1 else f"-{seen[base]}"
            entries.append(TocEntry(len(match.group(1)), title, f"{base}{suffix}"))
    return tuple(entries)


def build_toc(nb: NotebookNode) -> tuple[TocEntry, ...]:
    """Build deterministic table-of-contents entries for ATX Markdown headings."""
    return _toc_entries(nb)


def add_stable_heading_anchors(nb: NotebookNode) -> NotebookNode:
    """Return a deep copy with stable anchors immediately before Markdown headings."""
    notebook = copy.deepcopy(nb)
    entries = iter(_toc_entries(notebook))
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue

        def add_anchor(match: re.Match[str]) -> str:
            entry = next(entries)
            span = f'<span id="{entry.anchor}" class="heading-anchor" aria-hidden="true"></span>'
            return f"{span}\n{match.group(0)}"

        cell.source = _ATX_HEADING.sub(add_anchor, str(cell.source))
    return notebook
