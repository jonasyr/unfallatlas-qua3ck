from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from datetime import datetime
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile
from typing import Any

import nbformat
from jinja2 import Environment, FileSystemLoader, select_autoescape

from unfallatlas.presentation.assets import STATIC_DIR, write_atomic
from unfallatlas.presentation.metadata import BERLIN_TZ, source_sha256
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
_GIT_KEYS = {"commit", "short_commit", "branch", "dirty", "repo_name"}
_COUNT_KEYS = {
    "markdown",
    "code",
    "raw",
    "unexecuted_code",
    "executed_without_output",
    "code_with_output",
    "error_outputs",
}
_FINDING_KEYS = {"code", "severity", "message", "cell_index", "strict_blocker"}
_ASSET_KEYS = {"relative_path", "sha256", "size_bytes", "media_type", "kind", "cell_index"}


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


def _require_exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ManifestError(f"manifest {name} has missing or unknown fields")


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_relative_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ManifestError(f"manifest {name} must be a safe relative path")
    path = Path(value)
    windows_path = PureWindowsPath(value)
    if path.is_absolute() or windows_path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ManifestError(f"manifest {name} must be a safe relative path")
    return path.as_posix()


def _require_optional_index(value: Any, name: str) -> int | None:
    if value is not None and (not _is_integer(value) or value < 0):
        raise ManifestError(f"manifest {name} must be a non-negative integer or null")
    return value


def _validated_git(value: Any) -> dict[str, str | bool]:
    git = _require_mapping(value, "entry git")
    _require_exact_keys(git, _GIT_KEYS, "entry git")
    if not all(
        isinstance(git[key], str) for key in ("commit", "short_commit", "branch", "repo_name")
    ):
        raise ManifestError("manifest entry git text fields are invalid")
    if not isinstance(git["dirty"], bool):
        raise ManifestError("manifest entry git dirty must be a boolean")
    return git


def _validated_counts(value: Any) -> dict[str, int]:
    counts = _require_mapping(value, "entry cell_counts")
    _require_exact_keys(counts, _COUNT_KEYS, "entry cell_counts")
    if not all(_is_integer(item) and item >= 0 for item in counts.values()):
        raise ManifestError("manifest entry cell counts must be non-negative integers")
    return counts


def _validated_findings(value: Any) -> tuple[dict[str, str | int | bool | None], ...]:
    findings = _require_tuple_of_mappings(value, "entry findings")
    for finding in findings:
        _require_exact_keys(finding, _FINDING_KEYS, "entry finding")
        if not all(isinstance(finding[key], str) for key in ("code", "severity", "message")):
            raise ManifestError("manifest entry finding text fields are invalid")
        if finding["severity"] not in {"info", "warning", "error"}:
            raise ManifestError("manifest entry finding severity is invalid")
        _require_optional_index(finding["cell_index"], "entry finding cell_index")
        if not isinstance(finding["strict_blocker"], bool):
            raise ManifestError("manifest entry finding strict_blocker must be a boolean")
    return findings


def _validated_assets(value: Any) -> tuple[dict[str, str | int | None], ...]:
    assets = _require_tuple_of_mappings(value, "entry assets")
    for asset in assets:
        _require_exact_keys(asset, _ASSET_KEYS, "entry asset")
        asset["relative_path"] = _require_relative_path(
            asset["relative_path"], "entry asset relative_path"
        )
        if not all(isinstance(asset[key], str) for key in ("sha256", "media_type", "kind")):
            raise ManifestError("manifest entry asset text fields are invalid")
        if not _is_integer(asset["size_bytes"]) or asset["size_bytes"] < 0:
            raise ManifestError("manifest entry asset size_bytes must be a non-negative integer")
        _require_optional_index(asset["cell_index"], "entry asset cell_index")
    return assets


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
    _require_exact_keys(entry, required, "entry")
    string_fields = required - {"git", "cell_counts", "findings", "assets", "size_bytes"}
    if not all(isinstance(entry[field], str) for field in string_fields):
        raise ManifestError("manifest entry string fields are invalid")
    if entry["status"] not in {"ready", "wip", "placeholder", "invalid"}:
        raise ManifestError("manifest entry status is invalid")
    source = _require_relative_path(entry["source"], "entry source")
    output = _require_relative_path(entry["output"], "entry output")
    git = _validated_git(entry["git"])
    counts = _validated_counts(entry["cell_counts"])
    findings = _validated_findings(entry["findings"])
    assets = _validated_assets(entry["assets"])
    if not _is_integer(entry["size_bytes"]) or entry["size_bytes"] < 0:
        raise ManifestError("manifest entry size_bytes must be a non-negative integer")
    return ManifestEntry(
        source=source,
        output=output,
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


def _validated_entry(entry: ManifestEntry) -> ManifestEntry:
    primitives = json.loads(json.dumps(asdict(entry), ensure_ascii=False))
    return _entry_from_dict(primitives)


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
        if not _is_integer(root["schema_version"]) or root["schema_version"] != SCHEMA_VERSION:
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


def update_manifest(
    manifest: PresentationManifest,
    analysis: NotebookAnalysis,
    result: ExportResult,
    metadata: ExportMetadata,
    repo_root: Path,
    output_root: Path | None = None,
) -> PresentationManifest:
    """Return a manifest with the successful export inserted or replaced."""
    source = _relative_to(analysis.source, repo_root, "notebook source")
    export_root = (
        Path(output_root) if output_root is not None else repo_root / "reports" / "presentation"
    )
    entry = ManifestEntry(
        source=source,
        output=_relative_to(result.destination, export_root, "export destination"),
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


def _resolve_roots(
    notebooks_dir: Path | None,
    output_root: Path | None,
    repo_root: Path | None,
) -> tuple[Path, Path, Path]:
    if repo_root is not None:
        repository = Path(repo_root).resolve(strict=False)
        notebooks = (
            Path(notebooks_dir).resolve(strict=False)
            if notebooks_dir is not None
            else repository / "notebooks"
        )
    elif notebooks_dir is not None:
        notebooks = Path(notebooks_dir).resolve(strict=False)
        repository = notebooks.parent
    else:
        raise ValueError("notebooks_dir or repo_root is required")
    output = (
        Path(output_root).resolve(strict=False)
        if output_root is not None
        else repository / "reports" / "presentation"
    )
    return repository, notebooks, output


def _contained_path(root: Path, relative: str, description: str) -> Path:
    safe_relative = _require_relative_path(relative, description)
    resolved_root = root.resolve(strict=False)
    target = (resolved_root / safe_relative).resolve(strict=False)
    if not target.is_relative_to(resolved_root):
        raise ManifestError(f"manifest {description} escapes its root")
    return target


def _source_path(source: str, repo_root: Path, notebooks_dir: Path) -> Path:
    target = _contained_path(repo_root, source, "entry source")
    if not target.is_relative_to(notebooks_dir.resolve(strict=False)):
        raise ManifestError("manifest entry source is outside notebooks_dir")
    return target


def check_freshness(
    manifest: PresentationManifest,
    notebooks_dir: Path | None = None,
    *,
    output_root: Path | None = None,
    repo_root: Path | None = None,
) -> tuple[FreshnessResult, ...]:
    """Compare saved source-content hashes without considering execution outputs."""
    repository, notebooks, output = _resolve_roots(notebooks_dir, output_root, repo_root)
    results: list[FreshnessResult] = []
    for entry in sorted(manifest.entries, key=lambda item: item.source.casefold()):
        validated = _validated_entry(entry)
        source_path = _source_path(validated.source, repository, notebooks)
        output_path = _contained_path(output, validated.output, "entry output")
        if not source_path.exists():
            state = "orphaned"
        else:
            try:
                notebook = nbformat.read(source_path, as_version=4)
                state = "fresh" if source_sha256(notebook) == entry.source_sha256 else "stale"
            except (OSError, UnicodeError, nbformat.reader.NotJSONError):
                state = "invalid-source"
            if state == "fresh" and not output_path.is_file():
                state = "missing-export"
        if state not in _FRESHNESS_STATES:  # pragma: no cover - defensive invariant
            raise AssertionError(f"unknown freshness state: {state}")
        results.append(FreshnessResult(entry.source, entry.output, entry.status, state))
    return tuple(results)


def _shared_ui_href(output_root: Path, suffix: str) -> str:
    digest = sha256((STATIC_DIR / f"presentation{suffix}").read_bytes()).hexdigest()[:16]
    relative = Path("assets") / "ui" / f"presentation-{digest}{suffix}"
    target = _contained_path(output_root, relative.as_posix(), "shared UI asset")
    if not target.is_file():
        raise ManifestError(f"shared UI asset is absent: {relative.as_posix()}")
    return relative.as_posix()


def _render_index(
    manifest: PresentationManifest,
    output_root: Path,
    notebooks_dir: Path,
    repo_root: Path,
) -> str:
    freshness = check_freshness(
        manifest,
        notebooks_dir=notebooks_dir,
        output_root=output_root,
        repo_root=repo_root,
    )
    states = {result.source: result.state for result in freshness}
    rows = [
        {
            "entry": entry,
            "state": states[entry.source],
            "link": entry.output
            if _contained_path(output_root, entry.output, "entry output").is_file()
            else None,
        }
        for entry in manifest.entries
    ]
    buckets: dict[str, list[dict[str, Any]]] = {
        "Ready": [],
        "Work in progress": [],
        "Placeholder": [],
        "Invalid": [],
        "Outdated": [],
        "Orphaned": [],
    }
    status_sections = {
        "ready": "Ready",
        "wip": "Work in progress",
        "placeholder": "Placeholder",
        "invalid": "Invalid",
    }
    for row in rows:
        if row["state"] == "orphaned":
            section = "Orphaned"
        elif row["state"] != "fresh":
            section = "Outdated"
        else:
            section = status_sections[row["entry"].status]
        buckets[section].append(row)
    sections = tuple((title, items) for title, items in buckets.items() if items)
    generated_at_local = (
        datetime.fromisoformat(manifest.generated_at)
        .astimezone(BERLIN_TZ)
        .strftime("%d.%m.%Y, %H:%M:%S %Z")
        if manifest.generated_at
        else ""
    )
    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(("html", "j2")),
        keep_trailing_newline=True,
    )
    return environment.get_template("site_index.html.j2").render(
        manifest=manifest,
        generated_at_local=generated_at_local,
        sections=sections,
        style_href=_shared_ui_href(output_root, ".css"),
        script_src=_shared_ui_href(output_root, ".js"),
    )


def write_manifest_and_index(
    manifest: PresentationManifest,
    output_root: Path,
    *,
    notebooks_dir: Path | None = None,
    repo_root: Path | None = None,
) -> None:
    """Publish index first and replace the manifest last as the transaction marker."""
    output_root = Path(output_root).resolve(strict=False)
    if notebooks_dir is None and repo_root is None:
        if output_root.name != "presentation" or output_root.parent.name != "reports":
            raise ValueError("notebooks_dir or repo_root is required for a non-default output_root")
        repo_root = output_root.parent.parent
    repository, notebooks, output_root = _resolve_roots(notebooks_dir, output_root, repo_root)
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
        index = _render_index(manifest, output_root, notebooks, repository).encode("utf-8")
        write_atomic(output_root / "index.html", index)
        os.replace(temporary_manifest, manifest_path)
        temporary_manifest = None
    finally:
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)
