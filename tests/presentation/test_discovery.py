from pathlib import Path

import pytest
from conftest import write_notebook
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from unfallatlas.presentation.discovery import (
    classify_notebook_status,
    discover_notebooks,
    extract_notebook_title,
    resolve_explicit_notebooks,
)
from unfallatlas.presentation.models import NotebookStatus


def test_discovery_is_recursive_ipynb_only_and_sorted(tmp_path: Path) -> None:
    notebooks = tmp_path / "notebooks"
    write_notebook(notebooks / "10_Z.ipynb", [new_markdown_cell("# Z")])
    write_notebook(notebooks / "02_B.ipynb", [new_markdown_cell("# B")])
    write_notebook(notebooks / "nested" / "03_C.ipynb", [new_markdown_cell("# C")])
    (notebooks / "02_B.py").write_text("# mirror", encoding="utf-8")

    assert [p.relative_to(notebooks).as_posix() for p in discover_notebooks(notebooks)] == [
        "02_B.ipynb",
        "10_Z.ipynb",
        "nested/03_C.ipynb",
    ]


def test_discovery_uses_original_path_to_break_casefold_ties(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebooks = tmp_path / "notebooks"
    upper = notebooks / "A.ipynb"
    lower = notebooks / "a.ipynb"
    write_notebook(upper, [new_markdown_cell("# Upper")])
    write_notebook(lower, [new_markdown_cell("# Lower")])
    monkeypatch.setattr(Path, "rglob", lambda self, pattern: iter((lower, upper)))

    assert discover_notebooks(notebooks) == (upper, lower)


def test_resolve_explicit_notebooks_deduplicates_and_sorts_nested_paths(
    tmp_path: Path,
) -> None:
    notebooks = tmp_path / "notebooks"
    later = notebooks / "10_Z.ipynb"
    nested = notebooks / "future" / "03_C.ipynb"
    write_notebook(later, [new_markdown_cell("# Z")])
    write_notebook(nested, [new_markdown_cell("# C")])

    resolved = resolve_explicit_notebooks([later, nested, later], notebooks)

    assert resolved == (later.resolve(), nested.resolve())


def test_resolve_explicit_notebooks_uses_original_path_to_break_casefold_ties(
    tmp_path: Path,
) -> None:
    notebooks = tmp_path / "notebooks"
    upper = notebooks / "A.ipynb"
    lower = notebooks / "a.ipynb"
    write_notebook(upper, [new_markdown_cell("# Upper")])
    write_notebook(lower, [new_markdown_cell("# Lower")])

    resolved = resolve_explicit_notebooks([lower, upper], notebooks)

    assert resolved == (upper.resolve(), lower.resolve())


def test_resolve_explicit_notebooks_rejects_non_notebook(tmp_path: Path) -> None:
    notebooks = tmp_path / "notebooks"
    mirror = notebooks / "02_B.py"
    mirror.parent.mkdir(parents=True)
    mirror.write_text("# mirror", encoding="utf-8")

    with pytest.raises(ValueError, match=r"02_B\.py"):
        resolve_explicit_notebooks([mirror], notebooks)


def test_resolve_explicit_notebooks_rejects_path_outside_root(tmp_path: Path) -> None:
    notebooks = tmp_path / "notebooks"
    outside = tmp_path / "outside.ipynb"
    notebooks.mkdir()
    write_notebook(outside, [new_markdown_cell("# Outside")])

    with pytest.raises(ValueError, match=r"outside\.ipynb"):
        resolve_explicit_notebooks([outside], notebooks)


@pytest.mark.parametrize("status", ["wip", "placeholder"])
def test_explicit_presentation_status_takes_precedence(status: str) -> None:
    nb = new_notebook(
        cells=[new_code_cell("x = 1", execution_count=None)],
        metadata={"presentation": {"status": status}},
    )

    assert classify_notebook_status(nb) is NotebookStatus(status)


def test_explicit_ready_cannot_override_unexecuted_code() -> None:
    nb = new_notebook(
        cells=[new_code_cell("x = 1", execution_count=None)],
        metadata={"presentation": {"status": "ready"}},
    )

    assert classify_notebook_status(nb) is NotebookStatus.WIP


def test_invalid_explicit_presentation_status_uses_content_rule() -> None:
    nb = new_notebook(
        cells=[new_markdown_cell("# Complete narrative")],
        metadata={"presentation": {"status": "invalid"}},
    )

    assert classify_notebook_status(nb) is NotebookStatus.READY


def test_markdown_only_q_is_ready() -> None:
    nb = new_notebook(cells=[new_markdown_cell("# Q phase\nComplete narrative")])
    assert classify_notebook_status(nb) is NotebookStatus.READY


def test_empty_notebook_is_placeholder_not_ready() -> None:
    assert classify_notebook_status(new_notebook()) is NotebookStatus.PLACEHOLDER


def test_small_explicit_marker_is_placeholder() -> None:
    nb = new_notebook(cells=[new_markdown_cell("# C phase (TODO)")])
    assert classify_notebook_status(nb) is NotebookStatus.PLACEHOLDER


def test_many_cells_with_todo_comment_are_not_placeholder() -> None:
    cells = [new_markdown_cell("# A phase")] + [
        new_code_cell("# TODO note\nx = 1") for _ in range(3)
    ]
    nb = new_notebook(cells=cells)
    assert classify_notebook_status(nb) is NotebookStatus.WIP


def test_executed_code_is_ready() -> None:
    nb = new_notebook(cells=[new_code_cell("x = 1", execution_count=1)])
    assert classify_notebook_status(nb) is NotebookStatus.READY


def test_stored_error_output_forces_wip() -> None:
    error = {
        "output_type": "error",
        "ename": "ValueError",
        "evalue": "bad",
        "traceback": [],
    }
    nb = new_notebook(
        cells=[new_code_cell("raise ValueError", execution_count=1, outputs=[error])],
        metadata={"presentation": {"status": "ready"}},
    )

    assert classify_notebook_status(nb) is NotebookStatus.WIP


def test_title_uses_first_markdown_h1_and_strips_inline_punctuation() -> None:
    nb = new_notebook(
        cells=[
            new_code_cell("x = 1"),
            new_markdown_cell("Intro\n# **A³ Phase:** `Evaluation`!\nDetails"),
            new_markdown_cell("# Later title"),
        ]
    )

    assert extract_notebook_title(nb, "03_A3_Phase") == "A³ Phase Evaluation"


def test_title_falls_back_when_no_markdown_h1_exists() -> None:
    nb = new_notebook(cells=[new_markdown_cell("## Secondary heading")])
    assert extract_notebook_title(nb, "04_C_Phase") == "04_C_Phase"
