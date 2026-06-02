# Conventions

## Formatting & linting

- Line length: **100** (both ruff and black)
- Ruff rules: `E`, `F`, `I` (isort), `UP` (pyupgrade); `E501` ignored
- Target: Python 3.11
- Run `ruff check --fix` then `ruff format`; black is secondary

## Code style

- **No `print()`** in `src/` modules — use `logging`
- **Paths**: always `pathlib.Path`, never hardcoded strings
- Model artefacts → `data/processed/` (gitignored)
- Notebook outputs stripped by `nbstripout` in pre-commit

## Git / commits

- Conventional Commits enforced by commitizen (pre-commit `commit-msg` stage)
- Files in `docs/ConventionalCommitsGuide.md` for reference
- Large files blocked at >5 MB; private keys detected automatically

## ML-specific invariants

- Primary metric: **macro-F1** (class imbalance 1%/18%/81%)
- Always **stratified** splits; **chronological** split preferred (train≤2022, val=2023, test=2024)
- Never use random train/test splits on this time-series-adjacent data
