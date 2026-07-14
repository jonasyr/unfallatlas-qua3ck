import json
from datetime import UTC, datetime
from pathlib import Path

import nbformat
import pytest

from unfallatlas.presentation.manifest import (
    ManifestError,
    check_freshness,
    load_manifest,
    update_manifest,
    write_manifest_and_index,
)
from unfallatlas.presentation.metadata import snapshot_sha256, source_sha256
from unfallatlas.presentation.models import (
    AssetRecord,
    CellCounts,
    ExportMetadata,
    ExportResult,
    Finding,
    GitMetadata,
    NotebookAnalysis,
    NotebookStatus,
    PresentationManifest,
    Severity,
)


def _analysis(source: Path, *, status: NotebookStatus = NotebookStatus.READY) -> NotebookAnalysis:
    notebook = nbformat.read(source, as_version=4)
    findings = (Finding("saved-warning", Severity.WARNING, "Prüfen", 0),)
    return NotebookAnalysis(
        source=source,
        notebook=notebook,
        title=f"Titel {source.stem}",
        status=status,
        counts=CellCounts(0, 1, 0, 0, 0, 1, 0),
        findings=findings,
        snapshot_sha256=snapshot_sha256(notebook),
        source_sha256=source_sha256(notebook),
        output_bytes=3,
    )


def _add_entry(
    manifest: PresentationManifest,
    repo_root: Path,
    name: str,
    *,
    status: NotebookStatus = NotebookStatus.READY,
) -> PresentationManifest:
    source = repo_root / "notebooks" / f"{name}.ipynb"
    source.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("answer = 42", execution_count=1, outputs=[])]
    )
    nbformat.write(notebook, source)
    destination = repo_root / "reports" / "presentation" / "notebooks" / f"{name}.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("<html></html>", encoding="utf-8")
    result = ExportResult(
        source,
        destination,
        status,
        (Finding("saved-warning", Severity.WARNING, "Prüfen", 0),),
        destination.stat().st_size,
        None,
        (AssetRecord(Path("assets/notebooks/chart.png"), "abc", 17, "image/png", "image", 0),),
    )
    metadata = ExportMetadata(
        datetime(2026, 7, 15, 10, 30, tzinfo=UTC),
        "2026-07-15T12:30:00+02:00",
        GitMetadata("1234567890abcdef", "1234567890ab", "feature/export", True),
    )
    return update_manifest(manifest, _analysis(source, status=status), result, metadata, repo_root)


def test_manifest_serializes_schema_metadata_and_entries_deterministically(tmp_path: Path) -> None:
    manifest = _add_entry(PresentationManifest(), tmp_path, "zeta")
    manifest = _add_entry(manifest, tmp_path, "Ähre", status=NotebookStatus.WIP)

    output_root = tmp_path / "reports" / "presentation"
    write_manifest_and_index(manifest, output_root)

    raw = (output_root / "manifest.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    assert raw.endswith("\n")
    assert data["schema_version"] == 1
    assert [entry["source"] for entry in data["entries"]] == [
        "notebooks/zeta.ipynb",
        "notebooks/Ähre.ipynb",
    ]
    entry = data["entries"][0]
    assert entry["status"] == "ready"
    assert entry["git"]["dirty"] is True
    assert entry["findings"][0]["severity"] == "warning"
    assert entry["assets"][0]["size_bytes"] == 17
    assert load_manifest(output_root / "manifest.json") == manifest


def test_load_manifest_rejects_malformed_existing_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"schema_version": 1, "entries": "not-a-list"}', encoding="utf-8")

    with pytest.raises(ManifestError, match="manifest"):
        load_manifest(path)


def test_publication_does_not_overwrite_a_malformed_manifest(tmp_path: Path) -> None:
    output_root = tmp_path / "reports" / "presentation"
    output_root.mkdir(parents=True)
    manifest_path = output_root / "manifest.json"
    malformed = b"not json\n"
    manifest_path.write_bytes(malformed)

    with pytest.raises(ManifestError, match="manifest"):
        write_manifest_and_index(PresentationManifest(), output_root)

    assert manifest_path.read_bytes() == malformed
    assert not (output_root / "index.html").exists()


def test_freshness_ignores_outputs_then_detects_source_edit_and_deletion(tmp_path: Path) -> None:
    manifest = _add_entry(PresentationManifest(), tmp_path, "phase")
    source = tmp_path / "notebooks" / "phase.ipynb"
    notebook = nbformat.read(source, as_version=4)

    notebook.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="new")]
    notebook.cells[0].execution_count = 99
    nbformat.write(notebook, source)
    assert check_freshness(manifest, tmp_path / "notebooks")[0].state == "fresh"

    notebook.cells[0].source = "changed = True"
    nbformat.write(notebook, source)
    assert check_freshness(manifest, tmp_path / "notebooks")[0].state == "stale"

    source.unlink()
    assert check_freshness(manifest, tmp_path / "notebooks")[0].state == "orphaned"


def test_freshness_reports_missing_export_and_invalid_source(tmp_path: Path) -> None:
    manifest = _add_entry(PresentationManifest(), tmp_path, "phase")
    output = tmp_path / "reports" / "presentation" / "notebooks" / "phase.html"
    output.unlink()
    assert check_freshness(manifest, tmp_path / "notebooks")[0].state == "missing-export"

    (tmp_path / "notebooks" / "phase.ipynb").write_text("not json", encoding="utf-8")
    assert check_freshness(manifest, tmp_path / "notebooks")[0].state == "invalid-source"


def test_index_separates_ready_wip_placeholder_stale_and_orphaned(tmp_path: Path) -> None:
    manifest = PresentationManifest()
    for name, status in (
        ("ready", NotebookStatus.READY),
        ("work", NotebookStatus.WIP),
        ("stub", NotebookStatus.PLACEHOLDER),
        ("stale", NotebookStatus.READY),
        ("gone", NotebookStatus.READY),
    ):
        manifest = _add_entry(manifest, tmp_path, name, status=status)
    stale = tmp_path / "notebooks" / "stale.ipynb"
    notebook = nbformat.read(stale, as_version=4)
    notebook.cells[0].source = "edited = True"
    nbformat.write(notebook, stale)
    (tmp_path / "notebooks" / "gone.ipynb").unlink()
    missing_html = tmp_path / "reports" / "presentation" / "notebooks" / "work.html"
    missing_html.unlink()

    output_root = tmp_path / "reports" / "presentation"
    write_manifest_and_index(manifest, output_root)
    index = (output_root / "index.html").read_text(encoding="utf-8")

    assert "Bereit" in index
    assert "In Arbeit" in index
    assert "Platzhalter" in index
    assert "Veraltet" in index
    assert "Verwaist" in index
    assert 'href="notebooks/ready.html"' in index
    assert 'href="notebooks/work.html"' not in index
    assert "https://" not in index
    assert "http://" not in index
