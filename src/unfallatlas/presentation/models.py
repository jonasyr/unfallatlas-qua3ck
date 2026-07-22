import copy
import html
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from nbformat import NotebookNode

_ANCHOR_SPAN = re.compile(
    r'^<span id="[^"]+" class="heading-anchor" aria-hidden="true"></span>\n',
    re.MULTILINE,
)
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*)|[ \t]*)$")
_CLOSING_HASHES = re.compile(r"[ \t]+#+[ \t]*$")
_TABLE_TAG_ATTRIBUTES = {
    "table": frozenset({"aria-label", "aria-describedby"}),
    "caption": frozenset(),
    "colgroup": frozenset({"span"}),
    "col": frozenset({"span"}),
    "thead": frozenset(),
    "tbody": frozenset(),
    "tfoot": frozenset(),
    "tr": frozenset(),
    "th": frozenset({"id", "colspan", "rowspan", "headers", "scope", "abbr"}),
    "td": frozenset({"id", "colspan", "rowspan", "headers"}),
}
_URL_ATTRIBUTES = frozenset(
    {"href", "src", "data", "action", "formaction", "poster", "background", "xlink:href"}
)
_ACTIVE_STYLE = re.compile(r"(?:url\s*\(|expression\s*\(|@import|javascript:)", re.IGNORECASE)


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
class TocNode:
    level: int
    title: str
    anchor: str
    children: tuple["TocNode", ...]


@dataclass(frozen=True, slots=True)
class HtmlOutput:
    kind: str
    content: str


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
class ManifestEntry:
    source: str
    output: str
    title: str
    status: str
    exported_at: str
    exported_at_local: str
    git: dict[str, str | bool]
    snapshot_sha256: str
    source_sha256: str
    cell_counts: dict[str, int]
    findings: tuple[dict[str, str | int | bool | None], ...]
    assets: tuple[dict[str, str | int | None], ...]
    size_bytes: int


@dataclass(frozen=True, slots=True)
class PresentationManifest:
    schema_version: int = 1
    exporter_version: str = "1"
    generated_at: str = ""
    entries: tuple[ManifestEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class FreshnessResult:
    source: str
    output: str
    status: str
    state: str


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


def _heading_lines(source: str) -> tuple[tuple[int, int, str], ...]:
    headings: list[tuple[int, int, str]] = []
    fence_character: str | None = None
    fence_length = 0
    for line_index, line in enumerate(source.splitlines()):
        if fence_character is not None:
            closing = re.fullmatch(
                rf" {{0,3}}{re.escape(fence_character)}{{{fence_length},}}[ \t]*", line
            )
            if closing:
                fence_character = None
                fence_length = 0
            continue

        fence = _FENCE_OPEN.match(line)
        if fence and not (fence.group(1).startswith("`") and "`" in fence.group(2)):
            fence_character = fence.group(1)[0]
            fence_length = len(fence.group(1))
            continue

        heading = _ATX_HEADING.match(line)
        if not heading:
            continue
        markdown_title = _CLOSING_HASHES.sub("", heading.group(2) or "").strip()
        headings.append((line_index, len(heading.group(1)), _plain_heading_title(markdown_title)))
    return tuple(headings)


def _toc_entries(notebook: NotebookNode) -> tuple[TocEntry, ...]:
    allocated: set[str] = set()
    entries: list[TocEntry] = []
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        source = _ANCHOR_SPAN.sub("", str(cell.source))
        for _, level, title in _heading_lines(source):
            base = _slug(title)
            anchor = base
            suffix = 2
            while anchor in allocated:
                anchor = f"{base}-{suffix}"
                suffix += 1
            allocated.add(anchor)
            entries.append(TocEntry(level, title, anchor))
    return tuple(entries)


def build_toc(nb: NotebookNode) -> tuple[TocEntry, ...]:
    """Build deterministic table-of-contents entries for ATX Markdown headings."""
    return _toc_entries(nb)


def add_stable_heading_anchors(nb: NotebookNode) -> NotebookNode:
    """Return a deep copy with stable anchors immediately before Markdown headings."""
    notebook = copy.deepcopy(nb)
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            cell.source = _ANCHOR_SPAN.sub("", str(cell.source))
    entries = iter(_toc_entries(notebook))
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        heading_indexes = {line_index for line_index, _, _ in _heading_lines(str(cell.source))}
        rendered_lines: list[str] = []
        for line_index, line in enumerate(str(cell.source).splitlines(keepends=True)):
            if line_index in heading_indexes:
                entry = next(entries)
                rendered_lines.append(
                    f'<span id="{entry.anchor}" class="heading-anchor" aria-hidden="true"></span>\n'
                )
            rendered_lines.append(line)
        cell.source = "".join(rendered_lines)
    return notebook


@dataclass(slots=True)
class _MutableTocNode:
    entry: TocEntry
    children: list["_MutableTocNode"]


def nest_toc(entries: tuple[TocEntry, ...]) -> tuple[TocNode, ...]:
    """Convert flat heading entries into a semantic parent/child tree."""
    roots: list[_MutableTocNode] = []
    stack: list[_MutableTocNode] = []
    for entry in entries:
        node = _MutableTocNode(entry, [])
        while stack and stack[-1].entry.level >= entry.level:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)

    def freeze(node: _MutableTocNode) -> TocNode:
        return TocNode(
            node.entry.level,
            node.entry.title,
            node.entry.anchor,
            tuple(freeze(child) for child in node.children),
        )

    return tuple(freeze(root) for root in roots)


def classify_html_output(value: str) -> HtmlOutput:
    """Allow only sanitized passive table HTML; sandbox every other HTML output.

    pandas' DataFrame ``_repr_html_()`` wraps its ``<table>`` in a
    ``<div>...<style scoped>...</style><table>...</table></div>`` shell, so a
    literal single-top-level-``<table>`` check sends every DataFrame output to
    the sandboxed iframe fallback instead of the styled ``.table-scroll``
    path - the iframe is a separate document (no inherited page font/CSS) and
    always reserves a fixed min-height, which is why DataFrame tables render
    with default browser styling in an oversized box. Unwrap that specific,
    known-inert shell (drop the ``<style>`` block, keep only the table) before
    the table check, so real DataFrame output reaches the same sanitized,
    site-styled path as a hand-written table.
    """
    from bs4 import BeautifulSoup, NavigableString, Tag

    soup = BeautifulSoup(value, "html.parser")
    top_level = [
        item for item in soup.contents if not isinstance(item, NavigableString) or item.strip()
    ]
    if len(top_level) == 1 and isinstance(top_level[0], Tag) and top_level[0].name == "div":
        wrapper = top_level[0]
        wrapper_children = [
            item
            for item in wrapper.contents
            if not isinstance(item, NavigableString) or item.strip()
        ]
        non_style = [c for c in wrapper_children if not (isinstance(c, Tag) and c.name == "style")]
        if len(non_style) == 1 and isinstance(non_style[0], Tag) and non_style[0].name == "table":
            for style_tag in wrapper.find_all("style", recursive=False):
                style_tag.decompose()
            wrapper.unwrap()
            top_level = non_style

    if len(top_level) != 1 or not isinstance(top_level[0], Tag) or top_level[0].name != "table":
        return HtmlOutput("sandbox", value)

    for tag in soup.find_all(True):
        allowed_attributes = _TABLE_TAG_ATTRIBUTES.get(tag.name)
        if allowed_attributes is None:
            return HtmlOutput("sandbox", value)
        for attribute in tuple(tag.attrs):
            normalized = attribute.casefold()
            attribute_value = " ".join(tag.get_attribute_list(attribute))
            if normalized.startswith("on") or normalized in _URL_ATTRIBUTES:
                return HtmlOutput("sandbox", value)
            if normalized == "style" and _ACTIVE_STYLE.search(attribute_value):
                return HtmlOutput("sandbox", value)
            if normalized not in allowed_attributes:
                del tag.attrs[attribute]
    return HtmlOutput("table", str(soup))
