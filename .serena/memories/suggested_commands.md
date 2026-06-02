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
uv run ruff format src/                    # Format (preferred over black)
uv run black src/                          # Alternative formatter
uv run pytest                              # Run tests
pre-commit run --all-files                 # Run all pre-commit hooks
```

## Data

```bash
bash data/download_raw.sh                  # Download raw CSVs from BASt
```

## Package management

```bash
uv add <package>                           # Add dependency
uv sync                                    # Sync venv to lockfile
uv sync --all-extras                       # Include all extras (dev + geo)
uv sync --extra geo                        # Include geo extras only (h3, osmnx)
```
