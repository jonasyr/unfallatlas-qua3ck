#!/usr/bin/env python3
"""Prevent direct commits to generated notebook .py mirrors.

The .ipynb notebooks are the source of truth. Paired .py files exist only so
Serena/MCP and other symbolic tools can inspect notebook code.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def staged_files() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        check=True,
        text=True,
        capture_output=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    files = staged_files()

    changed_py_mirrors = [f for f in files if f.startswith("notebooks/") and f.endswith(".py")]

    if not changed_py_mirrors:
        return 0

    changed_ipynbs = {f for f in files if f.startswith("notebooks/") and f.endswith(".ipynb")}

    problems: list[str] = []

    for py_file in changed_py_mirrors:
        ipynb_file = str(Path(py_file).with_suffix(".ipynb"))
        if ipynb_file not in changed_ipynbs:
            problems.append(f"{py_file} changed without {ipynb_file}")

    if problems:
        print("ERROR: Direct edits to notebook .py mirrors are not allowed.\n")
        print("These files are generated mirrors for Serena/Jupytext only:")
        for problem in problems:
            print(f"  - {problem}")

        print(
            "\nEdit the matching .ipynb notebook instead, then regenerate mirrors:\n"
            "  uv run jupytext --sync notebooks/*.ipynb\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
