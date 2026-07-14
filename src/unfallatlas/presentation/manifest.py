from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import nbformat
from jinja2 import Environment, FileSystemLoader, select_autoescape

from unfallatlas.presentation.assets import write_atomic
from unfallatlas.presentation.metadata import source_sha256
from unfallatlas.presentation.models import (
    ExportMetadata,
    ExportResult,
    FreshnessResult,
    ManifestEntry,
    NotebookAnalysis,
    PresentationManifest,
)

SCHEMA_VERSION = 1
EXPORTER_VERSION = "1"
_FRESHNESS_STATES = frozenset({"fresh", "stale", "missing-export", "orphaned", "invalid-source"})


class ManifestError(ValueError):
    """Raised when an existing presentation manifest cannot be trusted."""


def _manifest_dict(manifest: PresentationManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "exporter_version": manifest.exporter_version,
        "generated_at": manifest.generated_at,
        "entries": [asdict(entry) for entry in manifest.entries],
    }


def _manifest_bytes(manifest: PresentationManifest) -> bytes:
    return (
        json.dumps(_manifest_dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"manifest {name} must be an object")
    return value


def _require_tuple_of_mappings(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ManifestError(f"manifest {name} must be a list of objects")
    return tuple(value)


def _entry_from_dict(value: Any) -> ManifestEntry:
    entry = _require_mapping(value, "entry")
    required = {
        "source",
        "output",
        "title",
        "status",
        "exported_at",
        "exported_at_local",
        "git",
        "snapshot_sha256",
        "source_sha256",
        "cell_counts",
        "findings",
        "assets",
        "size_bytes",
    }
    if set(entry) != required:
        raise ManifestError("manifest entry has missing or unknown fields")
    string_fields = required - {"git", "cell_counts", "findings", "assets", "size_bytes"}
    if not all(isinstance(entry[field], str) for field in string_fields):
        raise ManifestError("manifest entry string fields are invalid")
    if entry["status"] not in {"ready", "wip", "placeholder", "invalid"}:
        raise ManifestError("manifest entry status is invalid")
    git = _require_mapping(entry["git"], "entry git")
    counts = _require_mapping(entry["cell_counts"], "entry cell_counts")
    findings = _require_tuple_of_mappings(entry["findings"], "entry findings")
    assets = _require_tuple_of_mappings(entry["assets"], "entry assets")
    if not isinstance(entry["size_bytes"], int):
        raise ManifestError("manifest entry size_bytes must be an integer")
    if not all(isinstance(item, int) for item in counts.values()):
        raise ManifestError("manifest entry cell counts must be integers")
    return ManifestEntry(
        source=entry["source"],
        output=entry["output"],
        title=entry["title"],
        status=entry["status"],
        exported_at=entry["exported_at"],
        exported_at_local=entry["exported_at_local"],
        git=git,
        snapshot_sha256=entry["snapshot_sha256"],
        source_sha256=entry["source_sha256"],
        cell_counts=counts,
        findings=findings,
        assets=assets,
        size_bytes=entry["size_bytes"],
    )


def load_manifest(path: Path) -> PresentationManifest:
    """Load a versioned manifest, returning an empty manifest when none exists."""
    path = Path(path)
    if not path.exists():
        return PresentationManifest()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        root = _require_mapping(data, "root")
        if set(root) != {"schema_version", "exporter_version", "generated_at", "entries"}:
            raise ManifestError("manifest has missing or unknown fields")
        if root["schema_version"] != SCHEMA_VERSION:
            raise ManifestError(
                f"manifest schema version {root['schema_version']!r} is unsupported"
            )
        if not isinstance(root["exporter_version"], str) or not isinstance(
            root["generated_at"], str
        ):
            raise ManifestError("manifest metadata fields are invalid")
        if not isinstance(root["entries"], list):
            raise ManifestError("manifest entries must be a list")
        entries = tuple(_entry_from_dict(entry) for entry in root["entries"])
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, KeyError) as error:
        raise ManifestError(f"could not read manifest {path}: {error}") from error
    return PresentationManifest(
        schema_version=SCHEMA_VERSION,
        exporter_version=root["exporter_version"],
        generated_at=root["generated_at"],
        entries=tuple(sorted(entries, key=lambda entry: entry.source.casefold())),
    )


def _relative_to(path: Path, root: Path, description: str) -> str:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except ValueError as error:
        raise ValueError(f"{description} must be inside {root}") from error


def _output_relative(destination: Path, repo_root: Path) -> str:
    conventional_root = repo_root / "reports" / "presentation"
    try:
        return _relative_to(destination, conventional_root, "export destination")
    except ValueError:
        if destination.parent.name == "notebooks":
            return (Path("notebooks") / destination.name).as_posix()
        raise


def update_manifest(
    manifest: PresentationManifest,
    analysis: NotebookAnalysis,
    result: ExportResult,
    metadata: ExportMetadata,
    repo_root: Path,
) -> PresentationManifest:
    """Return a manifest with the successful export inserted or replaced."""
    source = _relative_to(analysis.source, repo_root, "notebook source")
    entry = ManifestEntry(
        source=source,
        output=_output_relative(result.destination, repo_root),
        title=analysis.title,
        status=result.status.value,
        exported_at=metadata.exported_at.isoformat(),
        exported_at_local=metadata.exported_at_local,
        git=asdict(metadata.git),
        snapshot_sha256=analysis.snapshot_sha256,
        source_sha256=analysis.source_sha256,
        cell_counts=asdict(analysis.counts),
        findings=tuple(asdict(finding) for finding in result.findings),
        assets=tuple(
            {
                **asdict(asset),
                "relative_path": asset.relative_path.as_posix(),
            }
            for asset in result.assets
        ),
        size_bytes=result.size_bytes,
    )
    retained = [existing for existing in manifest.entries if existing.source != source]
    entries = tuple(sorted((*retained, entry), key=lambda item: item.source.casefold()))
    return replace(
        manifest,
        schema_version=SCHEMA_VERSION,
        exporter_version=EXPORTER_VERSION,
        generated_at=metadata.exported_at.isoformat(),
        entries=entries,
    )


def _source_path(source: str, notebooks_dir: Path) -> Path:
    relative = Path(source)
    if relative.parts and relative.parts[0] == notebooks_dir.name:
        return notebooks_dir.parent / relative
    return notebooks_dir / relative


def _output_path(output: str, notebooks_dir: Path) -> Path:
    return notebooks_dir.parent / "reports" / "presentation" / output


def check_freshness(
    manifest: PresentationManifest, notebooks_dir: Path
) -> tuple[FreshnessResult, ...]:
    """Compare saved source-content hashes without considering execution outputs."""
    notebooks_dir = Path(notebooks_dir)
    results: list[FreshnessResult] = []
    for entry in sorted(manifest.entries, key=lambda item: item.source.casefold()):
        source_path = _source_path(entry.source, notebooks_dir)
        if not source_path.exists():
            state = "orphaned"
        else:
            try:
                notebook = nbformat.read(source_path, as_version=4)
                state = "fresh" if source_sha256(notebook) == entry.source_sha256 else "stale"
            except (OSError, UnicodeError, nbformat.reader.NotJSONError):
                state = "invalid-source"
            if state == "fresh" and not _output_path(entry.output, notebooks_dir).is_file():
                state = "missing-export"
        if state not in _FRESHNESS_STATES:  # pragma: no cover - defensive invariant
            raise AssertionError(f"unknown freshness state: {state}")
        results.append(FreshnessResult(entry.source, entry.output, entry.status, state))
    return tuple(results)


def _render_index(manifest: PresentationManifest, output_root: Path) -> str:
    notebooks_dir = output_root.parent.parent / "notebooks"
    states = {result.source: result.state for result in check_freshness(manifest, notebooks_dir)}
    rows = [
        {
            "entry": entry,
            "state": states[entry.source],
            "link": entry.output if (output_root / entry.output).is_file() else None,
        }
        for entry in manifest.entries
    ]
    sections = (
        (
            "Bereit",
            [row for row in rows if row["entry"].status == "ready" and row["state"] == "fresh"],
        ),
        ("In Arbeit", [row for row in rows if row["entry"].status == "wip"]),
        ("Platzhalter", [row for row in rows if row["entry"].status == "placeholder"]),
        (
            "Veraltet",
            [row for row in rows if row["state"] in {"stale", "missing-export", "invalid-source"}],
        ),
        ("Verwaist", [row for row in rows if row["state"] == "orphaned"]),
    )
    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(("html", "j2")),
        keep_trailing_newline=True,
    )
    return environment.get_template("site_index.html.j2").render(
        manifest=manifest,
        sections=tuple((title, items) for title, items in sections if items),
    )


def write_manifest_and_index(manifest: PresentationManifest, output_root: Path) -> None:
    """Publish index first and replace the manifest last as the transaction marker."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        load_manifest(manifest_path)
    temporary_manifest: Path | None = None
    try:
        with NamedTemporaryFile(dir=output_root, delete=False) as temporary:
            temporary_manifest = Path(temporary.name)
            temporary.write(_manifest_bytes(manifest))
            temporary.flush()
            os.fsync(temporary.fileno())
        index = _render_index(manifest, output_root).encode("utf-8")
        write_atomic(output_root / "index.html", index)
        os.replace(temporary_manifest, manifest_path)
        temporary_manifest = None
    finally:
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)
