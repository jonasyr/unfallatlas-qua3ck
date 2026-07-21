# Conventions

## Formatting & linting

- Line length: **100** (ruff; `[tool.black]` config still present in pyproject.toml but black is vestigial)
- Ruff rules: `E`, `F`, `I` (isort), `UP` (pyupgrade); `E501` ignored
- Target: Python 3.11
- Run `ruff check --fix` then `ruff format` — `ruff format` is the formatter actually enforced by pre-commit/CI

## Code style

- **No `print()`** in `src/` modules — use `logging`
- **Paths**: always `pathlib.Path`, never hardcoded strings
- Model artefacts → `data/processed/` (gitignored)
- Notebook outputs stripped by `nbstripout` in pre-commit

## Notebook policy

- `notebooks/*.ipynb` — source of truth; edit these, never the paired `.py` mirrors.
- `notebooks/*.py` — Jupytext/Serena mirrors for symbolic navigation only (read-only for agents).
- After editing a notebook: `uv run jupytext --sync notebooks/*.ipynb` then `serena project index`.
- Pre-commit hook `scripts/check_notebook_mirrors.py` (local hook, `check-notebook-mirrors` id) rejects commits where a `notebooks/*.py` mirror changed without its `.ipynb` counterpart.

## Git / commits

- Conventional Commits enforced by commitizen (pre-commit `commit-msg` stage)
- Files in `docs/project/ConventionalCommitsGuide.md` for reference
- Large files blocked at >5 MB; private keys detected automatically

## ML-specific invariants

- Primary metric: **macro-F1** (class imbalance 1%/18%/81%)
- Always **stratified** splits; **chronological** split preferred (train≤2022, val=2023, test=2024)
- Never use random train/test splits on this time-series-adjacent data

## Testing

- `pytest` marker `browser`: opt-in Playwright checks against exported presentation HTML
  (`tests/presentation/test_browser.py`); excluded by default via `-m "not browser"` in
  `pyproject.toml` addopts — run explicitly with `uv run pytest -m browser` (requires
  `presentation-test` extra installed)
