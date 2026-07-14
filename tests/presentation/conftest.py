from collections.abc import Sequence
from pathlib import Path
from typing import Any

import nbformat
from nbformat import NotebookNode


def write_notebook(
    path: Path,
    cells: Sequence[NotebookNode],
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.v4.new_notebook(cells=list(cells), metadata=metadata or {})
    nbformat.write(notebook, path)
