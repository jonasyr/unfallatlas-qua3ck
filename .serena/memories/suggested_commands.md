# Suggested Commands

All commands use `uv run` to stay inside the project venv.

## Development

```bash
uv run jupyter lab                          # Launch JupyterLab
uv run streamlit run app/streamlit_app.py   # Run Streamlit demo
```

## Notebooks

```bash
uv run jupytext --sync notebooks/*.ipynb    # Regenerate .py mirrors after editing .ipynb files
serena project index                        # Re-index Serena after mirror sync
```

Always run both commands together after any notebook change before committing.

## Quality

```bash
uv run ruff check src/ --fix               # Lint + auto-fix
uv run ruff format src/                    # Format (primary formatter; black is vestigial)
uv run pytest                              # Run tests (browser-marked tests excluded by default)
uv run pytest -m browser                   # Run opt-in Playwright checks against exported HTML
pre-commit install                         # One-time: activate git hooks for this clone
pre-commit run --all-files                 # Run all pre-commit hooks
```

## Data

```bash
bash data/download_raw.sh                  # Download raw CSVs from BASt
```

## Presentation export

```bash
uv sync --extra presentation                          # Install nbconvert/beautifulsoup4 export deps
uv run python scripts/export_notebooks.py --all        # Export all discovered notebooks to reports/presentation/
uv run python scripts/export_notebooks.py NOTEBOOK.ipynb  # Export a single notebook
uv run python scripts/export_notebooks.py --all --strict  # Fail on blocker findings (unexecuted cells, errors, etc.)
uv run python scripts/export_notebooks.py --check       # Check freshness of existing exports without rendering
```

## Package management

```bash
uv add <package>                           # Add dependency
uv sync                                    # Sync venv to lockfile
uv sync --all-extras                       # Include all extras (dev + geo)
uv sync --extra geo                        # Include geo extras only (h3, osmnx)
```
