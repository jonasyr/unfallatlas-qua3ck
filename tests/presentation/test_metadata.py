import subprocess
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from nbformat.v4 import new_code_cell, new_notebook, new_output

from unfallatlas.presentation.metadata import (
    build_export_metadata,
    read_git_metadata,
    snapshot_sha256,
    source_sha256,
)


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )


def test_source_hash_ignores_outputs_and_execution_metadata() -> None:
    first = new_notebook(
        cells=[
            new_code_cell(
                "x = 1",
                execution_count=1,
                outputs=[new_output("stream", name="stdout", text="one")],
                metadata={
                    "execution": {"iopub.status.busy": "2026-07-14T12:00:00Z"},
                    "collapsed": True,
                    "scrolled": True,
                    "tags": ["keep-me"],
                },
            )
        ],
        metadata={
            "widgets": {"state": "first"},
            "presentation": {"status": "ready"},
        },
    )
    second = deepcopy(first)
    second.cells[0].execution_count = None
    second.cells[0].outputs = []
    second.cells[0].metadata.pop("execution")
    second.cells[0].metadata.pop("collapsed")
    second.cells[0].metadata.pop("scrolled")
    second.metadata.pop("widgets")

    assert source_sha256(first) == source_sha256(second)
    assert snapshot_sha256(first) != snapshot_sha256(second)


def test_source_hash_changes_when_code_changes() -> None:
    first = new_notebook(cells=[new_code_cell("x = 1")])
    second = new_notebook(cells=[new_code_cell("x = 2")])

    assert source_sha256(first) != source_sha256(second)


def test_hashing_does_not_mutate_the_notebook() -> None:
    notebook = new_notebook(
        cells=[
            new_code_cell(
                "x = 1",
                execution_count=1,
                outputs=[new_output("stream", name="stdout", text="one")],
            )
        ],
        metadata={"widgets": {"state": "preserve"}},
    )
    original = deepcopy(notebook)

    snapshot_sha256(notebook)
    source_sha256(notebook)

    assert notebook == original


def test_read_git_metadata_detects_dirty_repository(tmp_path: Path) -> None:
    _run_git(tmp_path, "init")
    _run_git(tmp_path, "config", "user.email", "tests@example.invalid")
    _run_git(tmp_path, "config", "user.name", "Presentation Tests")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("committed\n", encoding="utf-8")
    _run_git(tmp_path, "add", "tracked.txt")
    _run_git(tmp_path, "commit", "-m", "test fixture")
    tracked.write_text("modified\n", encoding="utf-8")

    metadata = read_git_metadata(tmp_path)

    assert metadata.dirty is True
    assert metadata.commit != "unknown"
    assert metadata.short_commit == metadata.commit[:12]
    assert metadata.branch != "unknown"


def test_read_git_metadata_returns_unknown_outside_repository(tmp_path: Path) -> None:
    metadata = read_git_metadata(tmp_path)

    assert metadata.commit == "unknown"
    assert metadata.short_commit == "unknown"
    assert metadata.branch == "unknown"
    assert metadata.dirty is False


def test_build_export_metadata_uses_utc_seconds_and_local_readable_time(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 14, 15, 16, 17, 987654, tzinfo=UTC)

    metadata = build_export_metadata(tmp_path, now=now)

    assert metadata.exported_at == datetime(2026, 7, 14, 15, 16, 17, tzinfo=UTC)
    assert metadata.exported_at_local == metadata.exported_at.astimezone().isoformat(
        timespec="seconds"
    )
    assert metadata.git.commit == "unknown"
