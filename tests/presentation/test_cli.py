from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import nbformat
import pytest

from unfallatlas.presentation import cli
from unfallatlas.presentation.models import ExportResult


def _repo(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    return tmp_path


def _notebook(
    repo: Path,
    name: str,
    *,
    placeholder: bool = False,
    warning: bool = False,
) -> Path:
    if placeholder:
        cells = [nbformat.v4.new_markdown_cell("# TODO Platzhalter")]
    elif warning:
        cells = [
            nbformat.v4.new_markdown_cell(f"# {name}"),
            nbformat.v4.new_code_cell("1 + 1", execution_count=1, outputs=[]),
        ]
    else:
        cells = [nbformat.v4.new_markdown_cell(f"# {name}")]
    path = repo / "notebooks" / f"{name}.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nbformat.v4.new_notebook(cells=cells), path)
    return path


@pytest.fixture
def lightweight_publication(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    rendered: list[Path] = []

    def fake_render(
        analysis,
        metadata,
        output_root,
        *,
        repo_root=None,
        output_relative_path=None,
    ):
        relative = output_relative_path or Path(f"{analysis.source.stem}.html")
        destination = Path(output_root) / "notebooks" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("<html>fixture</html>", encoding="utf-8")
        rendered.append(analysis.source)
        return ExportResult(
            analysis.source,
            destination,
            analysis.status,
            analysis.findings,
            destination.stat().st_size,
            None,
        )

    def fake_publish(manifest, output_root, *, notebooks_dir=None, repo_root=None):
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": manifest.schema_version,
            "exporter_version": manifest.exporter_version,
            "generated_at": manifest.generated_at,
            "entries": [
                {
                    **asdict(entry),
                    "findings": list(entry.findings),
                    "assets": list(entry.assets),
                }
                for entry in manifest.entries
            ],
        }
        (output_root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        (output_root / "index.html").write_text("<html>index</html>", encoding="utf-8")

    monkeypatch.setattr(cli, "render_notebook", fake_render)
    monkeypatch.setattr(cli, "write_manifest_and_index", fake_publish)
    return rendered


def test_help_lists_documented_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["--help"])
    assert raised.value.code == 0
    output = capsys.readouterr().out
    for option in (
        "--all",
        "--check",
        "--output-dir",
        "--strict",
        "--include-placeholders",
        "--open",
    ):
        assert option in output


@pytest.mark.parametrize("arguments", [[], ["--all", "x.ipynb"], ["--all", "--check"]])
def test_parser_requires_exactly_one_selection(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)
    assert raised.value.code == 2


def test_one_notebook_exports_and_summary_has_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lightweight_publication: list[Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _repo(tmp_path)
    notebook = _notebook(repo, "one", warning=True)
    monkeypatch.chdir(repo)

    assert cli.main([str(notebook)]) == 0

    output = capsys.readouterr().out
    assert "one.ipynb" in output
    assert "wip" not in output
    assert "ready" in output
    assert "20 B" in output
    assert "1 info" in output
    assert "0 warnings" in output
    assert lightweight_publication == [notebook]


def test_all_skips_placeholders_unless_included(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lightweight_publication: list[Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _repo(tmp_path)
    ready = _notebook(repo, "ready")
    placeholder = _notebook(repo, "placeholder", placeholder=True)
    monkeypatch.chdir(repo)

    assert cli.main(["--all"]) == 0
    assert lightweight_publication == [ready]
    assert "skipped placeholder" in capsys.readouterr().out

    lightweight_publication.clear()
    assert cli.main(["--all", "--include-placeholders"]) == 0
    assert lightweight_publication == [placeholder, ready]


def test_normal_warning_succeeds_but_strict_blocker_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lightweight_publication: list[Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _repo(tmp_path)
    notebook = _notebook(repo, "warning")
    # A saved error output is a warning in normal mode and a strict blocker.
    nb = nbformat.read(notebook, as_version=4)
    nb.cells.append(
        nbformat.v4.new_code_cell(
            "raise ValueError()",
            execution_count=1,
            outputs=[nbformat.v4.new_output("error", ename="ValueError", evalue="", traceback=[])],
        )
    )
    nbformat.write(nb, notebook)
    monkeypatch.chdir(repo)

    assert cli.main([str(notebook)]) == 0
    lightweight_publication.clear()
    assert cli.main([str(notebook), "--strict"]) == 1
    assert lightweight_publication == []
    assert "strict validation blocked export" in capsys.readouterr().out


def test_invalid_explicit_path_is_selection_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    monkeypatch.chdir(repo)
    assert cli.main(["notebooks/missing.ipynb"]) == 2


def test_batch_continues_after_render_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lightweight_publication: list[Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _repo(tmp_path)
    bad = _notebook(repo, "a_bad")
    good = _notebook(repo, "b_good")
    original = cli.render_notebook

    def fail_first(
        analysis,
        metadata,
        output_root,
        *,
        repo_root=None,
        output_relative_path=None,
    ):
        result = original(
            analysis,
            metadata,
            output_root,
            repo_root=repo_root,
            output_relative_path=output_relative_path,
        )
        if analysis.source == bad:
            return replace(result, size_bytes=0, error="fixture render failed")
        return result

    monkeypatch.setattr(cli, "render_notebook", fail_first)
    monkeypatch.chdir(repo)

    assert cli.main(["--all"]) == 1
    assert lightweight_publication == [bad, good]
    output = capsys.readouterr().out
    assert "fixture render failed" in output
    assert "b_good.ipynb" in output


def test_batch_continues_after_unexpected_notebook_pipeline_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lightweight_publication: list[Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _repo(tmp_path)
    first = _notebook(repo, "a_first")
    second = _notebook(repo, "b_second")
    original = cli.build_export_metadata
    calls = 0

    def fail_once(repo_root):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("metadata unavailable")
        return original(repo_root)

    monkeypatch.setattr(cli, "build_export_metadata", fail_once)
    monkeypatch.chdir(repo)

    assert cli.main(["--all"]) == 1
    assert lightweight_publication == [second]
    output = capsys.readouterr().out
    assert first.name in output
    assert "metadata unavailable" in output
    assert second.name in output


def test_all_preserves_source_relative_paths_for_duplicate_nested_stems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lightweight_publication: list[Path],
) -> None:
    repo = _repo(tmp_path)
    first = _notebook(repo, "a/report")
    second = _notebook(repo, "b/report")
    monkeypatch.chdir(repo)

    assert cli.main(["--all"]) == 0

    output_root = repo / "reports" / "presentation"
    destinations = {
        output_root / "notebooks" / "a" / "report.html",
        output_root / "notebooks" / "b" / "report.html",
    }
    assert all(destination.is_file() for destination in destinations)
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert {entry["output"] for entry in manifest["entries"]} == {
        "notebooks/a/report.html",
        "notebooks/b/report.html",
    }
    assert lightweight_publication == [first, second]
    assert cli.main(["--check"]) == 0


def test_batch_publication_failure_marks_every_success_and_does_not_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lightweight_publication: list[Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _repo(tmp_path)
    _notebook(repo, "first")
    _notebook(repo, "second")
    opened: list[str] = []
    monkeypatch.setattr(
        cli,
        "write_manifest_and_index",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(cli.webbrowser, "open", lambda uri: opened.append(uri) or True)
    monkeypatch.chdir(repo)

    assert cli.main(["--all", "--open"]) == 1

    summaries = [line for line in capsys.readouterr().out.splitlines() if ".ipynb" in line]
    assert len(summaries) == 2
    assert all("failed — publication failed: disk full" in line for line in summaries)
    assert opened == []


def test_custom_output_directory_is_used_for_manifest_and_freshness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lightweight_publication: list[Path],
) -> None:
    repo = _repo(tmp_path)
    notebook = _notebook(repo, "custom")
    monkeypatch.chdir(repo / "notebooks")

    assert cli.main([str(notebook), "--output-dir", "../site"]) == 0
    assert (repo / "site" / "manifest.json").is_file()
    assert cli.main(["--check", "--output-dir", "../site"]) == 0

    nb = nbformat.read(notebook, as_version=4)
    nb.cells.append(nbformat.v4.new_markdown_cell("changed"))
    nbformat.write(nb, notebook)
    assert cli.main(["--check", "--output-dir", "../site"]) == 1


def test_check_never_renders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lightweight_publication: list[Path],
) -> None:
    repo = _repo(tmp_path)
    _notebook(repo, "one")
    monkeypatch.chdir(repo)
    assert cli.main(["--check"]) == 0
    assert lightweight_publication == []


def test_malformed_manifest_is_request_failure_not_selection_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lightweight_publication: list[Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _repo(tmp_path)
    output_root = repo / "reports" / "presentation"
    output_root.mkdir(parents=True)
    (output_root / "manifest.json").write_text("not json", encoding="utf-8")
    monkeypatch.chdir(repo)

    assert cli.main(["--check"]) == 1
    assert "could not read manifest" in capsys.readouterr().err


def test_open_uses_notebook_for_one_and_index_for_batch_only_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lightweight_publication: list[Path],
) -> None:
    repo = _repo(tmp_path)
    first = _notebook(repo, "first")
    _notebook(repo, "second")
    opened: list[str] = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda uri: opened.append(uri) or True)
    monkeypatch.chdir(repo)

    assert cli.main([str(first), "--open"]) == 0
    assert opened[-1].endswith("/notebooks/first.html")
    assert cli.main(["--all", "--open"]) == 0
    assert opened[-1].endswith("/index.html")

    monkeypatch.setattr(
        cli,
        "render_notebook",
        lambda analysis, metadata, output_root, **kwargs: ExportResult(
            analysis.source,
            Path(output_root) / "notebooks" / "failed.html",
            analysis.status,
            analysis.findings,
            0,
            "failed",
        ),
    )
    before = list(opened)
    assert cli.main([str(first), "--open"]) == 1
    assert opened == before


def test_open_does_nothing_when_all_selected_notebooks_are_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lightweight_publication: list[Path],
) -> None:
    repo = _repo(tmp_path)
    _notebook(repo, "placeholder", placeholder=True)
    opened: list[str] = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda uri: opened.append(uri) or True)
    monkeypatch.chdir(repo)

    assert cli.main(["--all", "--open"]) == 0
    assert opened == []


def test_find_repo_root_scans_parents(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    nested = repo / "a" / "b"
    nested.mkdir(parents=True)
    assert cli.find_repo_root(nested) == repo
