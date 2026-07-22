import json
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import nbformat
import pytest
from bs4 import BeautifulSoup

import unfallatlas.presentation.manifest as manifest_module
from unfallatlas.presentation.assets import STATIC_DIR, AssetStore, copy_shared_assets
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
    output_root: Path | None = None,
) -> PresentationManifest:
    source = repo_root / "notebooks" / f"{name}.ipynb"
    source.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("answer = 42", execution_count=1, outputs=[])]
    )
    nbformat.write(notebook, source)
    export_root = output_root or repo_root / "reports" / "presentation"
    destination = export_root / "notebooks" / f"{name}.html"
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
        GitMetadata(
            "1234567890abcdef", "1234567890ab", "feature/export", True, "unfallatlas-qua3ck"
        ),
    )
    return update_manifest(
        manifest,
        _analysis(source, status=status),
        result,
        metadata,
        repo_root,
        output_root=output_root,
    )


def _write_ui_assets(output_root: Path) -> None:
    store = AssetStore(output_root)
    for suffix, media_type, kind in (
        (".css", "text/css", "ui-style"),
        (".js", "text/javascript", "ui-script"),
    ):
        store.put_bytes(
            namespace="ui",
            stem="presentation",
            suffix=suffix,
            data=(STATIC_DIR / f"presentation{suffix}").read_bytes(),
            media_type=media_type,
            kind=kind,
            cell_index=None,
        )


def _manifest_data(manifest: PresentationManifest) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "exporter_version": manifest.exporter_version,
        "generated_at": manifest.generated_at,
        "entries": [asdict(entry) for entry in manifest.entries],
    }


def test_manifest_serializes_schema_metadata_and_entries_deterministically(tmp_path: Path) -> None:
    manifest = _add_entry(PresentationManifest(), tmp_path, "zeta")
    manifest = _add_entry(manifest, tmp_path, "Ähre", status=NotebookStatus.WIP)

    output_root = tmp_path / "reports" / "presentation"
    _write_ui_assets(output_root)
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


def test_custom_output_root_is_explicit_for_freshness_and_index(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    custom_root = tmp_path / "site"
    manifest = _add_entry(PresentationManifest(), repo_root, "phase", output_root=custom_root)
    _write_ui_assets(custom_root)

    fresh = check_freshness(
        manifest,
        notebooks_dir=repo_root / "notebooks",
        output_root=custom_root,
    )
    assert fresh[0].state == "fresh"

    write_manifest_and_index(
        manifest,
        custom_root,
        notebooks_dir=repo_root / "notebooks",
    )
    index = (custom_root / "index.html").read_text(encoding="utf-8")
    assert 'href="notebooks/phase.html"' in index


def test_index_requires_current_content_hashed_shared_ui_assets(tmp_path: Path) -> None:
    manifest = _add_entry(PresentationManifest(), tmp_path, "phase")
    output_root = tmp_path / "reports" / "presentation"

    with pytest.raises(ManifestError, match="shared UI asset"):
        write_manifest_and_index(manifest, output_root)

    records = copy_shared_assets(AssetStore(output_root))
    write_manifest_and_index(manifest, output_root)
    index = (output_root / "index.html").read_text(encoding="utf-8")
    ui_paths = {
        record.kind: record.relative_path.as_posix()
        for record in records
        if record.kind in {"ui-style", "ui-script"}
    }
    assert f'href="{ui_paths["ui-style"]}"' in index
    assert f'src="{ui_paths["ui-script"]}"' in index


def test_every_index_href_and_src_resolves_inside_output_root(tmp_path: Path) -> None:
    manifest = _add_entry(PresentationManifest(), tmp_path, "phase")
    output_root = tmp_path / "reports" / "presentation"
    _write_ui_assets(output_root)

    write_manifest_and_index(manifest, output_root)
    soup = BeautifulSoup((output_root / "index.html").read_text(encoding="utf-8"), "html.parser")

    targets = [tag.get("href") for tag in soup.find_all(href=True)]
    targets.extend(tag.get("src") for tag in soup.find_all(src=True))
    assert targets
    for target in targets:
        assert target is not None
        resolved = (output_root / target).resolve()
        assert resolved.is_relative_to(output_root.resolve())
        assert resolved.is_file()


def test_load_manifest_deeply_validates_nested_types_and_safe_paths(tmp_path: Path) -> None:
    manifest = _add_entry(PresentationManifest(), tmp_path, "phase")
    original = _manifest_data(manifest)
    invalid_values = (
        (("entries", 0, "git", "dirty"), 1),
        (("entries", 0, "cell_counts", "code"), True),
        (("entries", 0, "findings", 0, "cell_index"), True),
        (("entries", 0, "findings", 0, "strict_blocker"), 1),
        (("entries", 0, "assets", 0, "size_bytes"), True),
        (("entries", 0, "assets", 0, "cell_index"), True),
        (("entries", 0, "source"), "../outside.ipynb"),
        (("entries", 0, "output"), "/tmp/outside.html"),
        (("entries", 0, "assets", 0, "relative_path"), "assets/../outside.png"),
    )
    for path_parts, invalid in invalid_values:
        data = deepcopy(original)
        target = data
        for part in path_parts[:-1]:
            target = target[part]  # type: ignore[index]
        target[path_parts[-1]] = invalid  # type: ignore[index]
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ManifestError, match="manifest"):
            load_manifest(path)


def test_load_manifest_requires_exact_nested_keys(tmp_path: Path) -> None:
    manifest = _add_entry(PresentationManifest(), tmp_path, "phase")
    mutations = (
        lambda entry: entry["git"].update(extra="no"),
        lambda entry: entry["cell_counts"].pop("raw"),
        lambda entry: entry["findings"][0].update(extra="no"),
        lambda entry: entry["assets"][0].pop("kind"),
    )
    for mutate in mutations:
        data = _manifest_data(manifest)
        mutate(data["entries"][0])  # type: ignore[index]
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ManifestError, match="manifest"):
            load_manifest(path)


def test_in_memory_manifest_paths_are_checked_before_filesystem_access(tmp_path: Path) -> None:
    manifest = _add_entry(PresentationManifest(), tmp_path, "phase")
    unsafe = replace(manifest.entries[0], output="../outside.html")
    manifest = replace(manifest, entries=(unsafe,))

    with pytest.raises(ManifestError, match="relative path"):
        check_freshness(
            manifest,
            notebooks_dir=tmp_path / "notebooks",
            output_root=tmp_path / "reports" / "presentation",
        )


def test_index_separates_ready_wip_placeholder_stale_and_orphaned(tmp_path: Path) -> None:
    manifest = PresentationManifest()
    for name, status in (
        ("ready", NotebookStatus.READY),
        ("active", NotebookStatus.WIP),
        ("work", NotebookStatus.WIP),
        ("stub", NotebookStatus.PLACEHOLDER),
        ("stale", NotebookStatus.READY),
        ("stale-work", NotebookStatus.WIP),
        ("gone", NotebookStatus.READY),
        ("invalid", NotebookStatus.INVALID),
    ):
        manifest = _add_entry(manifest, tmp_path, name, status=status)
    stale = tmp_path / "notebooks" / "stale.ipynb"
    notebook = nbformat.read(stale, as_version=4)
    notebook.cells[0].source = "edited = True"
    nbformat.write(notebook, stale)
    stale_work = tmp_path / "notebooks" / "stale-work.ipynb"
    notebook = nbformat.read(stale_work, as_version=4)
    notebook.cells[0].source = "edited = True"
    nbformat.write(notebook, stale_work)
    (tmp_path / "notebooks" / "gone.ipynb").unlink()
    missing_html = tmp_path / "reports" / "presentation" / "notebooks" / "work.html"
    missing_html.unlink()

    output_root = tmp_path / "reports" / "presentation"
    _write_ui_assets(output_root)
    write_manifest_and_index(manifest, output_root)
    index = (output_root / "index.html").read_text(encoding="utf-8")

    assert "Bereit" in index
    assert "In Arbeit" in index
    assert "Platzhalter" in index
    assert "Ungültig" in index
    assert "Veraltet" in index
    assert "Verwaist" in index
    assert 'href="notebooks/ready.html"' in index
    assert 'href="notebooks/work.html"' not in index
    assert "https://" not in index
    assert "http://" not in index

    soup = BeautifulSoup(index, "html.parser")
    memberships = {
        section.find("h2").get_text(strip=True): [
            item.find(["a", "span"]).get_text(strip=True) for item in section.find_all("li")
        ]
        for section in soup.find_all("section")
    }
    expected_section = {
        "Titel ready": "Bereit",
        "Titel active": "In Arbeit",
        "Titel work": "Veraltet",
        "Titel stub": "Platzhalter",
        "Titel stale": "Veraltet",
        "Titel stale-work": "Veraltet",
        "Titel gone": "Verwaist",
        "Titel invalid": "Ungültig",
    }
    all_members = [title for titles in memberships.values() for title in titles]
    for title, section in expected_section.items():
        assert all_members.count(title) == 1
        assert title in memberships[section]


def test_manifest_is_replaced_after_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _add_entry(PresentationManifest(), tmp_path, "phase")
    output_root = tmp_path / "reports" / "presentation"
    _write_ui_assets(output_root)
    replacements: list[Path] = []
    original_replace = manifest_module.os.replace

    def recording_replace(source: Path, destination: Path) -> None:
        replacements.append(Path(destination))
        original_replace(source, destination)

    monkeypatch.setattr(manifest_module.os, "replace", recording_replace)
    write_manifest_and_index(manifest, output_root)

    assert replacements[-1] == output_root / "manifest.json"
    assert output_root / "index.html" in replacements[:-1]
