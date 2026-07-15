import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path

from nbformat import NotebookNode

from unfallatlas.presentation.models import NotebookStatus


def discover_notebooks(notebooks_dir: Path) -> tuple[Path, ...]:
    notebooks = (path for path in notebooks_dir.rglob("*.ipynb") if path.is_file())
    return tuple(
        sorted(
            notebooks,
            key=lambda path: (
                path.relative_to(notebooks_dir).as_posix().casefold(),
                path.relative_to(notebooks_dir).as_posix(),
            ),
        )
    )


def resolve_explicit_notebooks(paths: Sequence[Path], notebooks_dir: Path) -> tuple[Path, ...]:
    root = notebooks_dir.resolve()
    resolved: set[Path] = set()

    for path in paths:
        if path.suffix != ".ipynb":
            raise ValueError(f"Not a notebook: {path}")

        notebook = path.resolve(strict=True)
        if not notebook.is_relative_to(root):
            raise ValueError(f"Notebook is outside {notebooks_dir}: {path}")
        resolved.add(notebook)

    return tuple(
        sorted(
            resolved,
            key=lambda path: (
                path.relative_to(root).as_posix().casefold(),
                path.relative_to(root).as_posix(),
            ),
        )
    )


def classify_notebook_status(nb: NotebookNode) -> NotebookStatus:
    nonempty = [cell for cell in nb.cells if cell.get("source", "").strip()]
    if not nonempty:
        return NotebookStatus.PLACEHOLDER

    if any(
        output.get("output_type") == "error"
        for cell in nb.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    ):
        return NotebookStatus.WIP

    code = [cell for cell in nonempty if cell.cell_type == "code"]
    has_unexecuted_code = any(cell.get("execution_count") is None for cell in code)
    explicit = nb.metadata.get("presentation", {}).get("status")
    valid_explicit = {
        status.value for status in NotebookStatus if status is not NotebookStatus.INVALID
    }
    if explicit in valid_explicit:
        if explicit == NotebookStatus.READY.value and has_unexecuted_code:
            return NotebookStatus.WIP
        return NotebookStatus(explicit)

    first_text = "\n".join(cell.source for cell in nonempty[:2])
    marker = re.search(r"(?i)\b(?:TODO|TBD|Platzhalter)\b", first_text)
    if len(nonempty) <= 2 and not code and marker:
        return NotebookStatus.PLACEHOLDER
    if has_unexecuted_code:
        return NotebookStatus.WIP
    return NotebookStatus.READY


def extract_notebook_title(nb: NotebookNode, fallback: str) -> str:
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        match = re.search(r"^#\s+(.+?)\s*#*\s*$", cell.source, flags=re.MULTILINE)
        if match:
            title = "".join(
                " "
                if character == "`" or unicodedata.category(character).startswith("P")
                else character
                for character in match.group(1)
            )
            return " ".join(title.split()) or fallback
    return fallback
