# Task Completion Checklist

Run these before marking any coding task done:

1. **Lint**: `uv run ruff check src/ --fix`
2. **Format**: `uv run ruff format src/`
3. **Tests**: `uv run pytest`
4. **Notebooks** (if modified): ensure outputs stripped — `pre-commit run nbstripout` or commit triggers it automatically
5. **Pre-commit** (full check): `pre-commit run --all-files`

## Notes

- `pre-commit` handles ruff, nbstripout, commitizen, pyproject-fmt, detect-private-key, large-file check, nbqa-ruff in one pass.
- Tests are currently stubs (`tests/test_features.py` is nearly empty); a passing run with no failures is sufficient for now.
- Commit message must follow Conventional Commits format (enforced at commit-msg stage).
