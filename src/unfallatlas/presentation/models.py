from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from nbformat import NotebookNode


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
