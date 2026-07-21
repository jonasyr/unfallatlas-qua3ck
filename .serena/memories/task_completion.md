# Task Completion Checklist

Run these before marking any coding task done:

1. **Lint**: `uv run ruff check src/ --fix`
2. **Format**: `uv run ruff format src/` (primary formatter; black is vestigial — do not rely on `uv run black`)
3. **Tests**: `uv run pytest` (excludes `browser`-marked tests by default; run `uv run pytest -m browser` separately if presentation HTML changed)
4. **Notebooks** (if modified):
   - Sync mirrors: `uv run jupytext --sync notebooks/*.ipynb`
   - Re-index: `serena project index`
   - Outputs are stripped automatically by `nbstripout` at commit time; or run `pre-commit run nbstripout` manually.
   - Never commit a changed `notebooks/*.py` mirror without the matching `.ipynb` also being staged (enforced by `check-notebook-mirrors` hook).
5. **Presentation export** (if `src/unfallatlas/presentation/` or a notebook's saved outputs changed): re-run
   `uv run python scripts/export_notebooks.py --all --strict` and commit the refreshed `reports/presentation/`
   alongside the notebook.
6. **Pre-commit** (full check): `pre-commit run --all-files`

## Notes

- `pre-commit` handles ruff (`--fix` + `ruff-format`), nbstripout, commitizen, pyproject-fmt, detect-private-key, large-file check (≤5MB), nbqa-ruff, `check-notebook-mirrors` in one pass.
- Test suite is no longer a stub: ~25 files, ~5700 lines total, incl. `tests/presentation/` (12 files, ~4000 lines)
  and per-module coverage for osm/spatial/imbalance/svm/evaluate/etc. Only `tests/test_features.py` remains an
  empty stub — verify file size before assuming a module's tests are trivial.
- Commit message must follow Conventional Commits format (enforced at commit-msg stage).
