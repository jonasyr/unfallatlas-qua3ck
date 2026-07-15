# Notebook Presentation Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export saved Jupyter notebooks, without execution, into accessible offline HTML presentations with lossless lazy Plotly assets, validation, freshness metadata, and optional GitHub Pages publishing.

**Architecture:** A thin script delegates to focused modules under `src/unfallatlas/presentation/`. `nbformat` performs safe notebook reads and validation; a custom nbconvert template renders notebook structure; content-addressed assets keep HTML small and enable lazy `file://` Plotly loading. A JSON manifest is the source for freshness checks and the presentation index.

**Tech Stack:** Python 3.11+, pathlib, argparse, dataclasses, nbformat, nbconvert 7.17.x, Jinja2, Pygments, Beautiful Soup 4, Plotly.js from the installed Plotly package, MathJax 3.2.2 `tex-svg-full`, vanilla JavaScript/CSS, pytest, optional Python Playwright, uv, Git LFS, GitHub Pages Actions.

## Global Constraints

- Never execute notebook cells, start a kernel, import `ExecutePreprocessor`, retrain models, or access the full dataset during export.
- Treat `notebooks/*.ipynb` as source of truth and never edit `notebooks/*.py` mirrors directly.
- Preserve all pre-existing working-tree changes, especially the running A³ work and generated model artifacts.
- Support Python `>=3.11`, Windows, Linux, and macOS; use `pathlib.Path` for every filesystem path.
- Use argument-list `subprocess.run(..., shell=False)` for Git; never interpolate shell commands.
- Default output is `reports/presentation/`; the portable unit is the complete directory, not a lone HTML file.
- Normal mode exports safe notebooks despite warnings; strict mode uses the blocker matrix in the approved design.
- Keep the current 5-MiB pre-commit limit; use narrowly scoped Git LFS patterns instead of exemptions.
- Do not introduce Node, Vite, Webpack, Kaleido, or a custom notebook rendering engine.
- Write production modules with type annotations and logging rather than `print()`; terminal output belongs only in the CLI layer.
- Write failing tests before implementation, run the narrow test after each change, and commit only scoped task files.
- Do not publish or push externally during implementation; Pages deployment begins only after reviewed changes reach `main`.

## Planned File Map

Create:

```text
scripts/export_notebooks.py
src/unfallatlas/presentation/__init__.py
src/unfallatlas/presentation/models.py
src/unfallatlas/presentation/discovery.py
src/unfallatlas/presentation/metadata.py
src/unfallatlas/presentation/validation.py
src/unfallatlas/presentation/assets.py
src/unfallatlas/presentation/rendering.py
src/unfallatlas/presentation/manifest.py
src/unfallatlas/presentation/cli.py
src/unfallatlas/presentation/templates/notebook/conf.json
src/unfallatlas/presentation/templates/notebook/index.html.j2
src/unfallatlas/presentation/templates/site_index.html.j2
src/unfallatlas/presentation/static/presentation.css
src/unfallatlas/presentation/static/presentation.js
src/unfallatlas/presentation/vendor/mathjax-3.2.2-tex-svg-full.js
src/unfallatlas/presentation/vendor/MATHJAX-LICENSE.txt
tests/presentation/conftest.py
tests/presentation/test_models.py
tests/presentation/test_discovery.py
tests/presentation/test_metadata.py
tests/presentation/test_validation.py
tests/presentation/test_assets.py
tests/presentation/test_rendering.py
tests/presentation/test_manifest.py
tests/presentation/test_cli.py
tests/presentation/test_repository_config.py
tests/presentation/test_documentation.py
tests/presentation/test_browser.py
tests/presentation/fixtures/gallery.ipynb
docs/presentation-export.md
.github/workflows/pages.yml
```

Modify:

```text
pyproject.toml
uv.lock
.gitattributes
.gitignore
.github/workflows/ci.yml
README.md
```

Generated only after implementation and verification:

```text
reports/presentation/index.html
reports/presentation/manifest.json
reports/presentation/notebooks/01_Q_Phase.html
reports/presentation/notebooks/02_U_Phase.html
reports/presentation/assets/**
```

---

### Task 1: Presentation dependencies, package shell, and typed domain model

**Files:**
- Create: `src/unfallatlas/presentation/__init__.py`
- Create: `src/unfallatlas/presentation/models.py`
- Create: `tests/presentation/conftest.py`
- Create: `tests/presentation/test_models.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: `nbformat.NotebookNode`, `pathlib.Path`.
- Produces: `Severity`, `NotebookStatus`, `Finding`, `CellCounts`, `NotebookAnalysis`, `GitMetadata`, `ExportMetadata`, `AssetRecord`, `ExportResult`, and `BatchResult`.

- [ ] **Step 1: Add the failing model-contract tests**

Create `tests/presentation/test_models.py` with direct assertions for enum values, strict blockers, and batch success:

```python
from pathlib import Path

from unfallatlas.presentation.models import (
    BatchResult,
    ExportResult,
    Finding,
    NotebookStatus,
    Severity,
)


def test_finding_exposes_machine_code_and_strict_blocker() -> None:
    finding = Finding(
        code="UNEXECUTED_CELL",
        severity=Severity.WARNING,
        message="Code cell was never executed",
        cell_index=4,
        strict_blocker=True,
    )
    assert finding.code == "UNEXECUTED_CELL"
    assert finding.cell_index == 4
    assert finding.strict_blocker is True


def test_batch_result_succeeds_with_warnings() -> None:
    result = ExportResult(
        source=Path("notebooks/example.ipynb"),
        destination=Path("reports/presentation/notebooks/example.html"),
        status=NotebookStatus.READY,
        findings=(Finding("LARGE_OUTPUT", Severity.WARNING, "large"),),
        size_bytes=123,
        error=None,
    )
    assert BatchResult((result,)).successful is True
```

- [ ] **Step 2: Run the model tests and confirm the import failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/presentation/test_models.py -q`

Expected: FAIL because `unfallatlas.presentation.models` does not exist.

- [ ] **Step 3: Declare focused optional extras and implement the model types**

Add to `[project.optional-dependencies]` in `pyproject.toml`:

```toml
presentation = [
  "beautifulsoup4>=4.12,<5",
  "nbconvert>=7.17,<8",
]
presentation-test = [
  "playwright>=1.52,<2",
]
```

Implement frozen dataclasses in `models.py`. Use these exact public fields:

```python
class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class NotebookStatus(StrEnum):
    READY = "ready"
    WIP = "wip"
    PLACEHOLDER = "placeholder"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class Finding:
    code: str
    severity: Severity
    message: str
    cell_index: int | None = None
    strict_blocker: bool = False


@dataclass(frozen=True, slots=True)
class CellCounts:
    markdown: int
    code: int
    raw: int
    unexecuted_code: int
    executed_without_output: int
    code_with_output: int
    error_outputs: int


@dataclass(frozen=True, slots=True)
class NotebookAnalysis:
    source: Path
    notebook: NotebookNode
    title: str
    status: NotebookStatus
    counts: CellCounts
    findings: tuple[Finding, ...]
    snapshot_sha256: str
    source_sha256: str
    output_bytes: int

    @property
    def strict_blocked(self) -> bool:
        return any(finding.strict_blocker for finding in self.findings)
```

Also define `GitMetadata(commit, short_commit, branch, dirty)`,
`ExportMetadata(exported_at, exported_at_local, git)`,
`AssetRecord(relative_path, sha256, size_bytes, media_type, kind, cell_index)`,
`ExportResult(source, destination, status, findings, size_bytes, error, assets=())`, and
`BatchResult(results)` whose `successful` property is true when every result has
`error is None`.

Export the stable public types from `presentation/__init__.py`.

- [ ] **Step 4: Lock dependencies using uv**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv lock`

Expected: `uv.lock` records direct presentation and presentation-test requirements;
no unrelated package removals.

- [ ] **Step 5: Run the narrow tests and formatting checks**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_models.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/unfallatlas/presentation tests/presentation
```

Expected: tests PASS and Ruff reports no errors.

- [ ] **Step 6: Commit the model foundation**

```bash
git add pyproject.toml uv.lock src/unfallatlas/presentation/__init__.py src/unfallatlas/presentation/models.py tests/presentation/conftest.py tests/presentation/test_models.py
git commit -m "feat(presentation): add export domain model"
```

---

### Task 2: Deterministic discovery and notebook status classification

**Files:**
- Create: `src/unfallatlas/presentation/discovery.py`
- Create: `tests/presentation/test_discovery.py`
- Modify: `tests/presentation/conftest.py`

**Interfaces:**
- Consumes: `NotebookStatus` from Task 1.
- Produces: `discover_notebooks(notebooks_dir: Path) -> tuple[Path, ...]`,
  `resolve_explicit_notebooks(paths: Sequence[Path], notebooks_dir: Path) -> tuple[Path, ...]`,
  `classify_notebook_status(nb: NotebookNode) -> NotebookStatus`, and
  `extract_notebook_title(nb: NotebookNode, fallback: str) -> str`.

- [ ] **Step 1: Add reusable fixture builders and failing discovery tests**

In `conftest.py`, add `write_notebook(path, cells, metadata=None)` using
`nbformat.v4.new_notebook` and `nbformat.write`. In `test_discovery.py`, cover:

```python
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


def test_markdown_only_q_is_ready() -> None:
    nb = new_notebook(cells=[new_markdown_cell("# Q phase\nComplete narrative")])
    assert classify_notebook_status(nb) is NotebookStatus.READY


def test_small_explicit_marker_is_placeholder() -> None:
    nb = new_notebook(cells=[new_markdown_cell("# C phase (TODO)")])
    assert classify_notebook_status(nb) is NotebookStatus.PLACEHOLDER


def test_many_cells_with_todo_comment_are_not_placeholder() -> None:
    cells = [new_markdown_cell("# A phase")] + [new_code_cell("# TODO note\nx = 1") for _ in range(3)]
    nb = new_notebook(cells=cells)
    assert classify_notebook_status(nb) is NotebookStatus.WIP
```

Also test explicit `metadata.presentation.status`, duplicate explicit inputs, a `.py`
input, a path outside the notebook root, and a future nested `.ipynb` file.

- [ ] **Step 2: Run discovery tests and confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_discovery.py -q`

Expected: FAIL because discovery functions are undefined.

- [ ] **Step 3: Implement discovery and the conservative status rule**

Use `Path.resolve(strict=True)` and `Path.is_relative_to(notebooks_dir.resolve())` for
explicit paths. Reject non-`.ipynb` inputs with a `ValueError` containing the path.
Sort by `relative_to(notebooks_dir).as_posix().casefold()` and return tuples.

Classification must:

```python
explicit = nb.metadata.get("presentation", {}).get("status")
if explicit in {status.value for status in NotebookStatus if status is not NotebookStatus.INVALID}:
    return NotebookStatus(explicit)

nonempty = [cell for cell in nb.cells if cell.get("source", "").strip()]
code = [cell for cell in nonempty if cell.cell_type == "code"]
first_text = "\n".join(cell.source for cell in nonempty[:2])
marker = re.search(r"(?i)\b(?:TODO|TBD|Platzhalter)\b", first_text)
if len(nonempty) <= 2 and not code and marker:
    return NotebookStatus.PLACEHOLDER
if any(cell.get("execution_count") is None for cell in code):
    return NotebookStatus.WIP
return NotebookStatus.READY
```

Title extraction uses the first Markdown ATX h1, strips inline Markdown punctuation,
and falls back to the filename stem.

- [ ] **Step 4: Run tests and lint**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_discovery.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/unfallatlas/presentation/discovery.py tests/presentation/test_discovery.py
```

Expected: PASS.

- [ ] **Step 5: Commit discovery**

```bash
git add src/unfallatlas/presentation/discovery.py tests/presentation/conftest.py tests/presentation/test_discovery.py
git commit -m "feat(presentation): discover and classify notebooks"
```

---

### Task 3: Canonical notebook hashes, export clock, and safe Git metadata

**Files:**
- Create: `src/unfallatlas/presentation/metadata.py`
- Create: `tests/presentation/test_metadata.py`

**Interfaces:**
- Consumes: `GitMetadata`, `ExportMetadata` from Task 1.
- Produces: `snapshot_sha256(nb: NotebookNode) -> str`,
  `source_sha256(nb: NotebookNode) -> str`,
  `read_git_metadata(repo_root: Path) -> GitMetadata`, and
  `build_export_metadata(repo_root: Path, now: datetime | None = None) -> ExportMetadata`.

- [ ] **Step 1: Write failing hash and Git tests**

Tests must prove that output stripping does not change the source hash, source edits do,
snapshot output edits do, dirty Git is detected in a temporary repository, and missing
Git returns `unknown` rather than raising:

```python
def test_source_hash_ignores_outputs_and_execution_counts() -> None:
    first = new_notebook(cells=[new_code_cell("x = 1", execution_count=1, outputs=[new_output("stream", name="stdout", text="one")])])
    second = deepcopy(first)
    second.cells[0].execution_count = None
    second.cells[0].outputs = []
    assert source_sha256(first) == source_sha256(second)
    assert snapshot_sha256(first) != snapshot_sha256(second)


def test_source_hash_changes_when_code_changes() -> None:
    first = new_notebook(cells=[new_code_cell("x = 1")])
    second = new_notebook(cells=[new_code_cell("x = 2")])
    assert source_sha256(first) != source_sha256(second)
```

Create a temporary Git repository with local test user configuration, one commit, then
modify a file and assert `dirty is True` and the branch/commit fields are populated.

- [ ] **Step 2: Run the metadata tests and confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_metadata.py -q`

Expected: FAIL because metadata functions do not exist.

- [ ] **Step 3: Implement deterministic canonicalization and Git fallbacks**

Deep-copy notebooks before canonicalization. Snapshot JSON uses
`nbformat.writes(nb, version=4)` normalized through `json.loads` then
`json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.

For the source hash, remove from every code cell `outputs` and set
`execution_count=None`; remove notebook `metadata.widgets` and execution-only cell
metadata keys `execution`, `collapsed`, and `scrolled`. Do not remove source,
attachments, tags, or presentation status.

Implement a private `_git(args, root)` using:

```python
subprocess.run(
    ["git", *args],
    cwd=root,
    check=False,
    capture_output=True,
    text=True,
    shell=False,
)
```

Use `rev-parse HEAD`, `rev-parse --short=12 HEAD`, `branch --show-current`, and
`status --porcelain=v1`. Catch `OSError`; failed commands yield `unknown`, while an
empty porcelain result means clean. Export time uses aware UTC, `astimezone()` for the
readable local time, and seconds precision.

- [ ] **Step 4: Run tests and lint**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_metadata.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/unfallatlas/presentation/metadata.py tests/presentation/test_metadata.py
```

Expected: PASS.

- [ ] **Step 5: Commit metadata support**

```bash
git add src/unfallatlas/presentation/metadata.py tests/presentation/test_metadata.py
git commit -m "feat(presentation): record reproducible export metadata"
```

---

### Task 4: Structured notebook, execution, MIME, resource, and offline validation

**Files:**
- Create: `src/unfallatlas/presentation/validation.py`
- Create: `tests/presentation/test_validation.py`
- Modify: `tests/presentation/conftest.py`

**Interfaces:**
- Consumes: `NotebookAnalysis`, `Finding`, `CellCounts`, status/title helpers, and hash helpers.
- Produces: `read_and_validate_notebook(source: Path, repo_root: Path) -> NotebookAnalysis`,
  `scan_runtime_references(html: str, notebook_dir: Path, repo_root: Path, cell_index: int) -> tuple[Finding, ...]`,
  and `NotebookReadError`.

- [ ] **Step 1: Write the full validation matrix as failing tests**

Create small tests for Markdown-only, executed text, unexecuted code, executed-no-output,
Error output, monotonic/duplicate counts, HTML fallback, unsupported-only MIME, widget
without state, local image present/missing, traversal, external image/script/iframe,
normal hyperlink, Plotly OpenStreetMap tiles, single payload over 5 MiB, and total
payload over 100 MiB. Representative assertions:

```python
def codes(analysis: NotebookAnalysis) -> set[str]:
    return {finding.code for finding in analysis.findings}


def test_executed_outputless_cell_is_info_not_strict(tmp_path: Path) -> None:
    path = write_notebook(tmp_path / "notebooks/setup.ipynb", [new_code_cell("x = 1", execution_count=1)])
    analysis = read_and_validate_notebook(path, tmp_path)
    finding = next(item for item in analysis.findings if item.code == "EXECUTED_NO_OUTPUT")
    assert finding.severity is Severity.INFO
    assert finding.strict_blocker is False


def test_missing_local_image_blocks_strict(tmp_path: Path) -> None:
    path = write_notebook(tmp_path / "notebooks/missing.ipynb", [new_markdown_cell("![chart](missing.png)")])
    analysis = read_and_validate_notebook(path, tmp_path)
    assert "MISSING_LOCAL_ASSET" in codes(analysis)
    assert analysis.strict_blocked is True


def test_normal_external_hyperlink_is_not_runtime_dependency(tmp_path: Path) -> None:
    path = write_notebook(tmp_path / "notebooks/link.ipynb", [new_markdown_cell("[Source](https://example.org/paper)")])
    analysis = read_and_validate_notebook(path, tmp_path)
    assert "EXTERNAL_RUNTIME_RESOURCE" not in codes(analysis)
```

For the map test, create a Plotly MIME bundle whose layout contains
`mapbox.style="open-street-map"` and assert `EXTERNAL_MAP_TILES` is a strict blocker.

- [ ] **Step 2: Run validation tests and confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_validation.py -q`

Expected: FAIL because validation is not implemented.

- [ ] **Step 3: Implement safe reading and execution-state analysis**

Read UTF-8 with `nbformat.read(source, as_version=4)` and call `nbformat.validate`.
Wrap decoding, JSON, schema, and OS errors in `NotebookReadError` with the source path.

Count only non-empty code. Emit these exact codes:

```text
UNEXECUTED_CELL
EXECUTED_NO_OUTPUT
ERROR_OUTPUT
NON_MONOTONIC_EXECUTION
DUPLICATE_EXECUTION_COUNT
MISSING_LOCAL_ASSET
UNSAFE_LOCAL_ASSET
EXTERNAL_RUNTIME_RESOURCE
EXTERNAL_MAP_TILES
UNSUPPORTED_MIME
WIDGET_STATE_MISSING
LARGE_OUTPUT
VERY_LARGE_NOTEBOOK_OUTPUT
PLACEHOLDER_NOTEBOOK
WIP_NOTEBOOK
```

Unexecuted, errors, non-monotonic/duplicate counts, WIP, missing/unsafe resources,
external runtime resources, unsupported MIME, and missing widget state set
`strict_blocker=True`. Size findings do not.

- [ ] **Step 4: Implement MIME fallback and secure resource scanning**

Supported MIME priority is Plotly, widget-with-state, HTML, SVG, PNG, JPEG, LaTeX,
plain text, and JavaScript/active HTML in sandbox. Data Wrangler is supported only
when `text/html` or `text/plain` is also present.

Use Beautiful Soup to inspect Markdown-renderable raw HTML and output `text/html`.
Treat `img/src`, `script/src`, `link[rel=stylesheet]/href`, `iframe/src`,
`source/src`, `video/src`, `audio/src`, and CSS `url(...)` as runtime resources.
Resolve local paths against `source.parent`, call `resolve(strict=False)`, and require
`candidate.is_relative_to(repo_root.resolve())`. Ignore fragment, `mailto:`, and
normal anchor hyperlinks.

Inspect Plotly layout for Mapbox/open-street-map styles, map layers, and external
images. Calculate payload size from compact UTF-8 JSON, using 5 MiB and 100 MiB
thresholds.

- [ ] **Step 5: Run validation tests and lint**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_validation.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/unfallatlas/presentation/validation.py tests/presentation/test_validation.py
```

Expected: PASS.

- [ ] **Step 6: Commit validation**

```bash
git add src/unfallatlas/presentation/validation.py tests/presentation/conftest.py tests/presentation/test_validation.py
git commit -m "feat(presentation): validate notebook snapshots"
```

---

### Task 5: Content-addressed asset store and lossless Plotly preparation

**Files:**
- Create: `src/unfallatlas/presentation/assets.py`
- Create: `tests/presentation/test_assets.py`
- Create: `src/unfallatlas/presentation/vendor/MATHJAX-LICENSE.txt`
- Create: `src/unfallatlas/presentation/vendor/mathjax-3.2.2-tex-svg-full.js`

**Interfaces:**
- Consumes: validated `NotebookAnalysis` and `AssetRecord`.
- Produces: `AssetStore`, `PreparedNotebook`,
  `prepare_notebook_assets(analysis: NotebookAnalysis, store: AssetStore) -> PreparedNotebook`, and
  `copy_shared_assets(store: AssetStore) -> tuple[AssetRecord, ...]`.

- [ ] **Step 1: Write failing asset tests**

Test deterministic digest paths, byte-identical deduplication, image extraction,
Plotly JS registration, source notebook immutability, one Plotly runtime, MathJax
presence, and relative hrefs. The key Plotly assertion is:

```python
prepared = prepare_notebook_assets(analysis, AssetStore(output_root))
record = next(asset for asset in prepared.assets if asset.kind == "plotly")
payload = (output_root / record.relative_path).read_text(encoding="utf-8")
assert payload.startswith('window.UnfallatlasPresentation.registerPlotlyPayload(')
assert '"data"' in payload
assert analysis.notebook.cells[0].outputs[0].data["application/vnd.plotly.v1+json"] == original
```

Also test that a second equal Plotly bundle returns the same digest asset and stable
payload key but separate chart IDs, and that image filenames use the correct `.png`,
`.jpg`, or `.svg` suffix.

- [ ] **Step 2: Run asset tests and confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_assets.py -q`

Expected: FAIL because asset preparation is undefined.

- [ ] **Step 3: Implement atomic, content-addressed asset writes**

`AssetStore` owns `output_root` and provides:

```python
def put_bytes(
    self,
    *,
    namespace: str,
    stem: str,
    suffix: str,
    data: bytes,
    media_type: str,
    kind: str,
    cell_index: int | None,
) -> AssetRecord:
    digest = hashlib.sha256(data).hexdigest()
    relative = Path("assets") / namespace / f"{stem}-{digest[:16]}{suffix}"
    write_atomic(self.output_root / relative, data)
    return AssetRecord(relative, digest, len(data), media_type, kind, cell_index)
```

`write_atomic` uses `NamedTemporaryFile(dir=target.parent, delete=False)`, flushes,
calls `os.fsync`, closes, then `os.replace`; it unlinks the temporary path on errors.

- [ ] **Step 4: Prepare Plotly and binary outputs on a deep copy**

For each Plotly output, serialize the MIME dict with compact JSON. Derive a stable
`payload_key` from its full SHA-256 and write one deduplicated asset per payload:

```javascript
window.UnfallatlasPresentation.registerPlotlyPayload("<payload-key>", <plotly-json>);
```

Add to the copied output metadata:

```json
{
  "unfallatlas_presentation": {
    "kind": "plotly",
    "chart_id": "<notebook-key>-cell-<n>-output-<m>",
    "payload_key": "plotly-<full-sha256>",
    "asset_href": "../assets/notebooks/<notebook-key>/<file>.js",
    "size_bytes": 62042876
  }
}
```

For PNG/JPEG/SVG, decode or encode the MIME payload, write an asset, and add equivalent
`kind=image` metadata. Preserve text/HTML fallbacks. Never mutate `analysis.notebook`.

- [ ] **Step 5: Add pinned MathJax and shared Plotly copying**

Fetch the official fixed MathJax file and its license with:

```bash
curl -fsSL https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg-full.js \
  -o src/unfallatlas/presentation/vendor/mathjax-3.2.2-tex-svg-full.js
curl -fsSL https://raw.githubusercontent.com/mathjax/MathJax/3.2.2/LICENSE \
  -o src/unfallatlas/presentation/vendor/MATHJAX-LICENSE.txt
sha256sum src/unfallatlas/presentation/vendor/mathjax-3.2.2-tex-svg-full.js
```

Expected SHA-256:

```text
a4354ff94fd868aea0cc6eaaa79a57fda0588646fc46ee3700a349ee0a11cbe6
```

Expected size: 2,275,113 bytes. Confirm `MATHJAX-LICENSE.txt` names Apache License 2.0
and add a two-line provenance header naming MathJax 3.2.2 and the exact upstream URL.

Locate Plotly with `importlib.resources.files("plotly") / "package_data" / "plotly.min.js"`,
copy it once to `assets/vendor/plotly-<plotly.__version__>.min.js`, and record its hash.
Copy MathJax once to `assets/vendor/mathjax-3.2.2-tex-svg-full.js`. Copy the packaged
CSS and JavaScript byte-for-byte to content-addressed files under `assets/ui/`; return
all four shared records so the renderer never depends on package filesystem paths.

- [ ] **Step 6: Run asset tests and lint**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_assets.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/unfallatlas/presentation/assets.py tests/presentation/test_assets.py
```

Expected: PASS, with no external Plotly or MathJax URLs in produced assets.

- [ ] **Step 7: Commit asset preparation**

```bash
git add src/unfallatlas/presentation/assets.py src/unfallatlas/presentation/vendor tests/presentation/test_assets.py
git commit -m "feat(presentation): externalize rich notebook assets"
```

---

### Task 6: Custom nbconvert template, stable TOC, accessible controls, and print CSS

**Files:**
- Create: `src/unfallatlas/presentation/templates/notebook/conf.json`
- Create: `src/unfallatlas/presentation/templates/notebook/index.html.j2`
- Create: `src/unfallatlas/presentation/static/presentation.css`
- Create: `src/unfallatlas/presentation/static/presentation.js`
- Create: `tests/presentation/test_rendering.py`

**Interfaces:**
- Consumes: prepared output metadata from Task 5 and nbconvert's `basic`/`classic` blocks.
- Produces: `build_toc(nb: NotebookNode) -> tuple[TocEntry, ...]`,
  `add_stable_heading_anchors(nb: NotebookNode) -> NotebookNode`, and the custom
  template contract expected by Task 7.

- [ ] **Step 1: Write failing structural rendering tests**

Use a small in-memory notebook and a temporary template render. Parse the result with
Beautiful Soup and assert:

```python
assert soup.select_one("header.presentation-header")
assert soup.select_one('nav[aria-label="Inhaltsverzeichnis"]')
assert [item["href"] for item in soup.select(".toc-link")] == [
    "#grossen-qualitat",
    "#grossen-qualitat-2",
]
assert soup.select_one("details.code-cell:not([open])")
assert soup.select_one("details.output-cell[open]")
assert {button["data-action"] for button in soup.select("button[data-action]")} >= {
    "show-all-code",
    "hide-all-code",
    "show-all-output",
    "hide-all-output",
    "print",
}
assert soup.select_one('[aria-live="polite"]')
assert "https://cdn" not in html
```

Also assert a Plotly output creates a `.plotly-output[data-asset]` placeholder, tables
are inside `.table-scroll`, text is inside `.text-output`, Error output uses
`.error-output`, and metadata displays commit, dirty state, time, counts, and warnings.

- [ ] **Step 2: Run rendering tests and confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_rendering.py -q`

Expected: FAIL because template helpers and files do not exist.

- [ ] **Step 3: Implement stable heading IDs and TOC entries**

Define `TocEntry(level: int, title: str, anchor: str)` in `models.py`. Slug creation
uses Unicode NFKC, casefolding, whitespace-to-hyphen, removal of characters other than
Unicode letters/numbers/underscore/hyphen, an explicit `ß -> ss` normalization, and
duplicate suffixes `-2`, `-3`, and so on.

On a deep copy, insert `<span id="<anchor>" class="heading-anchor" aria-hidden="true"></span>`
immediately before each ATX Markdown heading. Preserve the original heading Markdown
so inline emphasis/code continues through nbconvert. Build the TOC from the same pass.

- [ ] **Step 4: Implement the nbconvert template contract**

`conf.json` must inherit from `basic`, enable HTML MIME, and keep the Pygments CSS
preprocessor. `index.html.j2` extends `basic/index.html.j2`, supplies a full document
shell, and overrides these blocks: `header`, `codecell`, `input_group`, `output_group`,
`markdowncell`, `data_priority`, `stream_stdout`, `stream_stderr`, `error`, and `footer`.

In `data_priority`, check `output.metadata.unfallatlas_presentation.kind` before
nbconvert's standard MIME ordering. Plotly markup is:

```html
<div class="plotly-output"
     id="{{ chart_id }}"
     data-chart-id="{{ chart_id }}"
     data-payload-key="{{ payload_key }}"
     data-asset="{{ asset_href }}"
     data-size-bytes="{{ size_bytes }}"
     role="img"
     aria-label="Interaktive Plotly-Grafik">
  <p class="output-placeholder">Grafik wird beim Öffnen geladen.</p>
</div>
```

Image metadata renders a responsive `<img>` with a local href. HTML containing script
or iframe markup is emitted through a sandboxed `srcdoc` iframe; ordinary table HTML
is wrapped in `.table-scroll`. Preserve Pygments-generated code HTML and copy/paste.

- [ ] **Step 5: Implement accessible progressive-enhancement JavaScript**

`presentation.js` must expose one namespace and no dependencies:

```javascript
window.UnfallatlasPresentation = (() => {
  const plotlyPayloads = new Map();
  const plotlyWaiters = new Map();
  const loadedScripts = new Map();

  function registerPlotlyPayload(payloadKey, payload) {
    plotlyPayloads.set(payloadKey, payload);
    const waiter = plotlyWaiters.get(payloadKey);
    if (waiter) waiter(payload);
  }

  function loadScript(src) {
    if (!loadedScripts.has(src)) {
      loadedScripts.set(src, new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = src;
        script.onload = resolve;
        script.onerror = () => reject(new Error(`Lokales Asset konnte nicht geladen werden: ${src}`));
        document.head.append(script);
      }));
    }
    return loadedScripts.get(src);
  }

  function waitForPayload(id) {
    if (plotlyPayloads.has(id)) return Promise.resolve(plotlyPayloads.get(id));
    return new Promise((resolve) => plotlyWaiters.set(id, resolve));
  }

  async function loadPlotly(container) {
    if (container.dataset.loaded === "true") return;
    try {
      await loadScript(document.body.dataset.plotlyRuntime);
      const pendingPayload = waitForPayload(container.dataset.payloadKey);
      await loadScript(container.dataset.asset);
      const payload = await pendingPayload;
      container.replaceChildren();
      await window.Plotly.newPlot(
        container,
        payload.data,
        payload.layout,
        {responsive: true, displaylogo: false},
      );
      container.dataset.loaded = "true";
    } catch (error) {
      const message = document.createElement("p");
      message.className = "output-load-error";
      message.textContent = error instanceof Error ? error.message : "Grafik konnte nicht geladen werden.";
      container.replaceChildren(message);
    }
  }

  return {registerPlotlyPayload, loadPlotly};
})();
```

Complete the module with:

- native-details global toggles and synchronized `aria-expanded`;
- `sessionStorage` keys namespaced by the body `data-snapshot-sha256`;
- loading Plotly when output details open or an `IntersectionObserver` approaches;
- active TOC section tracking;
- mobile TOC drawer with focus return and Escape close;
- table/output expansion buttons;
- back-to-top behavior;
- print button that awaits all Plotly loads before `window.print()`;
- a `beforeprint` best-effort loader;
- an `aria-live` status region;
- no failure path that hides Markdown or existing output text.

- [ ] **Step 6: Implement responsive and print CSS**

Use CSS custom properties for a restrained light palette. Required selectors include:
`.presentation-shell`, `.presentation-header`, `.metadata-grid`, `.toc`, `.notebook-main`,
`.code-cell`, `.output-cell`, `.table-scroll`, `.text-output`, `.error-output`,
`.plotly-output`, `.focus-visible`, and `.back-to-top`.

Set a readable content maximum around 1100px, sticky TOC only above 1100px, horizontal
table overflow, 32rem table and 28rem text maximum heights, sticky `thead`, code
horizontal scrolling with wrapping disabled by default, responsive images/SVG, and
visible `:focus-visible` outlines. Under 760px, use a full-width non-overlapping TOC
drawer.

`@media print` hides navigation/buttons, forces `details > *` to display, hides
summaries, removes height limits, constrains figures, repeats table headers, applies
`break-after: avoid` to headings, and uses `print-color-adjust: economy`.
`@media (prefers-reduced-motion: reduce)` disables smooth scrolling/transitions.

- [ ] **Step 7: Run rendering tests and static scans**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_rendering.py -q
rg -n "https?://|cdn|unpkg|jsdelivr" src/unfallatlas/presentation/templates src/unfallatlas/presentation/static
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/unfallatlas/presentation tests/presentation/test_rendering.py
```

Expected: tests PASS; URL scan has no runtime CDN reference; Ruff passes.

- [ ] **Step 8: Commit the presentation UI**

```bash
git add src/unfallatlas/presentation/models.py src/unfallatlas/presentation/templates src/unfallatlas/presentation/static tests/presentation/test_rendering.py
git commit -m "feat(presentation): add accessible notebook template"
```

---

### Task 7: End-to-end renderer and atomic notebook publication

**Files:**
- Create: `src/unfallatlas/presentation/rendering.py`
- Modify: `tests/presentation/test_rendering.py`

**Interfaces:**
- Consumes: `NotebookAnalysis`, metadata, prepared assets, template helpers.
- Produces: `render_notebook(analysis: NotebookAnalysis, metadata: ExportMetadata, output_root: Path) -> ExportResult` and `PresentationHTMLExporter`.

- [ ] **Step 1: Write failing integration and atomicity tests**

Create an executed notebook with Markdown, text, table, SVG, and Plotly MIME. Assert one
HTML file, local assets, no source mutation, no execution, correct title/metadata, and
all asset hrefs exist. Test atomic preservation by pre-writing `old html`, monkeypatching
the final `os.replace` to raise, and asserting the old target remains byte-identical.

Also patch `subprocess.run` and notebook execution APIs to fail if called, proving the
renderer does not execute code.

- [ ] **Step 2: Run renderer integration tests and confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_rendering.py -q`

Expected: new integration tests FAIL because `render_notebook` is missing.

- [ ] **Step 3: Implement the exporter configuration**

Subclass `HTMLExporter` only to register filters and template paths; do not build a
renderer from notebook JSON. Configure:

```python
exporter = PresentationHTMLExporter(
    template_name="notebook",
    extra_template_basedirs=[str(package_templates_root)],
)
exporter.exclude_input_prompt = True
exporter.exclude_output_prompt = True
exporter.embed_images = False
```

Pass all presentation data under `resources["presentation"]`: title, status, findings,
counts, TOC, metadata, snapshot hash, source path, shared asset hrefs, and asset map.
Suppress only the known custom Plotly MIME unsupported warning because the template
handles it; do not blanket-suppress nbconvert warnings.

- [ ] **Step 4: Implement staged publication order**

The renderer must:

1. create output directories;
2. copy shared assets atomically;
3. deep-copy and anchor Markdown;
4. externalize rich assets;
5. call `from_notebook_node`;
6. encode UTF-8 and atomically replace `notebooks/<stem>.html`;
7. return exact size and all `AssetRecord` values.

If any step fails, return or raise an error without deleting previous output. Convert
exceptions to `ExportResult.error` at the orchestration boundary, not deep inside asset
functions.

- [ ] **Step 5: Run integration tests and lint**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_rendering.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/unfallatlas/presentation/rendering.py tests/presentation/test_rendering.py
```

Expected: PASS.

- [ ] **Step 6: Commit renderer**

```bash
git add src/unfallatlas/presentation/rendering.py tests/presentation/test_rendering.py
git commit -m "feat(presentation): render notebooks without execution"
```

---

### Task 8: Manifest, freshness checks, orphan handling, and site index

**Files:**
- Create: `src/unfallatlas/presentation/manifest.py`
- Create: `src/unfallatlas/presentation/templates/site_index.html.j2`
- Create: `tests/presentation/test_manifest.py`

**Interfaces:**
- Consumes: `NotebookAnalysis`, `ExportMetadata`, `ExportResult`.
- Produces: `load_manifest(path: Path) -> PresentationManifest`,
  `update_manifest(manifest: PresentationManifest, analysis: NotebookAnalysis, result: ExportResult, metadata: ExportMetadata, repo_root: Path) -> PresentationManifest`,
  `check_freshness(manifest: PresentationManifest, notebooks_dir: Path) -> tuple[FreshnessResult, ...]`,
  `write_manifest_and_index(manifest: PresentationManifest, output_root: Path) -> None`.

- [ ] **Step 1: Write failing manifest and freshness tests**

Test JSON schema version, deterministic entry order, output-stripped source staying
fresh, code edit becoming stale, missing source becoming orphaned, warning/status
serialization, dirty Git fields, asset sizes, and ready/WIP/placeholder index sections.

```python
fresh = check_freshness(manifest, notebooks_dir)
assert fresh[0].state == "fresh"

nb.cells[0].source = "changed = True"
nbformat.write(nb, source_path)
assert check_freshness(manifest, notebooks_dir)[0].state == "stale"

source_path.unlink()
assert check_freshness(manifest, notebooks_dir)[0].state == "orphaned"
```

- [ ] **Step 2: Run manifest tests and confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_manifest.py -q`

Expected: FAIL because manifest support is undefined.

- [ ] **Step 3: Implement versioned manifest dataclasses and JSON conversion**

Add `ManifestEntry`, `PresentationManifest`, and `FreshnessResult` to `models.py` with
JSON-safe primitive fields. Use schema version `1`. Serialize with UTF-8,
`ensure_ascii=False`, `indent=2`, `sort_keys=True`, and a trailing newline.

Manifest entries contain source/output paths relative to repository/output root,
title/status, exported times, Git fields, hashes, cell counts, findings, assets, and
size. Sort entries by `source.casefold()`.

Malformed existing manifests raise `ManifestError`; do not silently overwrite them.

- [ ] **Step 4: Implement freshness and index generation**

Freshness reads current notebooks through nbformat and compares `source_sha256`.
States are exactly `fresh`, `stale`, `missing-export`, `orphaned`, and `invalid-source`.

Render `site_index.html.j2` with ready entries first; WIP, placeholder, stale, and
orphaned entries appear in clearly labelled secondary sections. Do not link missing
HTML. Reuse local UI CSS/JS only; include no CDN. Atomically write manifest first to a
temporary path, render index from that in-memory manifest, atomically replace index,
then replace manifest last so it is the transaction marker.

- [ ] **Step 5: Run manifest tests and lint**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_manifest.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/unfallatlas/presentation/manifest.py tests/presentation/test_manifest.py
```

Expected: PASS.

- [ ] **Step 6: Commit manifest and index**

```bash
git add src/unfallatlas/presentation/models.py src/unfallatlas/presentation/manifest.py src/unfallatlas/presentation/templates/site_index.html.j2 tests/presentation/test_manifest.py
git commit -m "feat(presentation): track snapshot freshness"
```

---

### Task 9: Cross-platform CLI, batch continuation, strict mode, and summaries

**Files:**
- Create: `src/unfallatlas/presentation/cli.py`
- Create: `scripts/export_notebooks.py`
- Create: `tests/presentation/test_cli.py`

**Interfaces:**
- Consumes: discovery, validation, metadata, rendering, and manifest functions.
- Produces: `build_parser() -> argparse.ArgumentParser`,
  `run_export(args: argparse.Namespace, repo_root: Path) -> BatchResult`, and
  `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write failing CLI tests**

Cover `--help`, mutually exclusive selection, one notebook, `--all`, placeholder skip,
`--include-placeholders`, normal warnings exit 0, strict blocker exit 1, invalid path
exit 2, one failure not stopping the next notebook, `--output-dir`, `--check` fresh and
stale, and `--open` target selection.

Use direct `main([...])` calls and monkeypatch `webbrowser.open`; do not spawn costly
project processes. Assert the summary contains notebook name, status, size, finding
counts, and error reason.

- [ ] **Step 2: Run CLI tests and confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_cli.py -q`

Expected: FAIL because CLI modules do not exist.

- [ ] **Step 3: Implement parser and repository-root detection**

Parser usage must require exactly one of positional `notebooks`, `--all`, or `--check`.
Add `--output-dir`, `--strict`, `--include-placeholders`, and `--open`.

Resolve repository root with `find_repo_root(start: Path | None = None)`: first scan
`Path.cwd().resolve()` and its parents for both `pyproject.toml` and `.git`; if that
fails, scan the installed module path and its parents. Raise a concise CLI error if no
root is found. This accepts invocation from the repository root or a subdirectory
without hard-coded parent counts. The script contains only:

```python
#!/usr/bin/env python3
from unfallatlas.presentation.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Implement batch orchestration and exit rules**

For every selected path:

1. read and validate;
2. skip placeholders unless included;
3. if strict and blocked, record a failed result and continue;
4. build metadata and render;
5. update the in-memory manifest;
6. continue after errors.

Write index/manifest once after all successful notebook renders. Normal warnings do not
change exit 0. Any failed requested notebook changes exit to 1. Argparse retains exit
2. `--check` performs no rendering and returns 1 for any non-fresh requested entry.

Terminal formatting uses plain Unicode text with a no-color-safe layout; production
modules log but do not print. `--open` runs only after successful publication.

- [ ] **Step 5: Run CLI tests, help, and lint**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_cli.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation python scripts/export_notebooks.py --help
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check scripts/export_notebooks.py src/unfallatlas/presentation/cli.py tests/presentation/test_cli.py
```

Expected: tests PASS; help lists all documented options; Ruff passes.

- [ ] **Step 6: Commit CLI**

```bash
git add scripts/export_notebooks.py src/unfallatlas/presentation/cli.py tests/presentation/test_cli.py
git commit -m "feat(presentation): add notebook export CLI"
```

---

### Task 10: Git LFS boundaries, generated-file ignores, CI, and GitHub Pages

**Files:**
- Modify: `.gitattributes`
- Modify: `.gitignore`
- Modify: `.github/workflows/ci.yml`
- Create: `.github/workflows/pages.yml`
- Create: `tests/presentation/test_repository_config.py`

**Interfaces:**
- Consumes: output layout and CLI from Tasks 7–9.
- Produces: committable large assets, CI dependency coverage, and static Pages deployment.

- [ ] **Step 1: Write failing repository-configuration tests**

Read configuration as text and assert:

```python
def test_presentation_large_assets_are_lfs_scoped(repo_root: Path) -> None:
    attrs = (repo_root / ".gitattributes").read_text(encoding="utf-8")
    assert "reports/presentation/assets/notebooks/** filter=lfs" in attrs
    assert "reports/presentation/assets/vendor/** filter=lfs" in attrs


def test_pages_workflow_never_executes_notebooks(repo_root: Path) -> None:
    workflow = (repo_root / ".github/workflows/pages.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "actions/upload-pages-artifact@v4" in workflow
    assert "actions/deploy-pages@v4" in workflow
    assert "jupyter" not in workflow.lower()
    assert "pytest" not in workflow.lower()
```

Also assert the Pages path filter, `lfs: true`, minimum permissions, concurrency,
presentation extra in CI, and that `reports/presentation/` is not globally ignored.

- [ ] **Step 2: Run configuration tests and confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_repository_config.py -q`

Expected: FAIL because the config is not present.

- [ ] **Step 3: Add narrow LFS rules and temporary-file ignores**

Append to `.gitattributes`:

```gitattributes
reports/presentation/assets/notebooks/** filter=lfs diff=lfs merge=lfs -text
reports/presentation/assets/vendor/** filter=lfs diff=lfs merge=lfs -text
```

Append to `.gitignore` only:

```gitignore
# Interrupted atomic presentation writes
reports/presentation/**/.tmp-*
```

Do not ignore the presentation directory, HTML, index, manifest, or UI assets.

- [ ] **Step 4: Update CI installation and add Pages workflow**

Change CI installation to:

```yaml
- name: Install dependencies
  run: uv sync --extra dev --extra geo --extra presentation
```

Create Pages workflow with:

```yaml
name: Deploy presentations to Pages

on:
  push:
    branches: [main]
    paths:
      - "reports/presentation/**"
      - ".github/workflows/pages.yml"
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    permissions:
      contents: read
      pages: write
      id-token: write
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
        with:
          lfs: true
      - name: Validate presentation artifact
        run: |
          test -f reports/presentation/index.html
          test -f reports/presentation/manifest.json
          test "$(du -sm reports/presentation | cut -f1)" -lt 1024
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v4
        with:
          path: reports/presentation
      - name: Deploy
        id: deployment
        uses: actions/deploy-pages@v4
```

These action majors were checked against the official GitHub Pages action documentation
on 2026-07-14: checkout v6, configure-pages v5, upload-pages-artifact v4, and
deploy-pages v4. Recheck them only if implementation occurs substantially later; any
version change must update workflow and test together.

- [ ] **Step 5: Run config tests and inspect attributes**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_repository_config.py -q
git check-attr filter -- reports/presentation/assets/notebooks/example.js reports/presentation/assets/vendor/plotly.min.js
```

Expected: tests PASS; both example paths report `filter: lfs`.

- [ ] **Step 6: Commit repository integration**

```bash
git add .gitattributes .gitignore .github/workflows/ci.yml .github/workflows/pages.yml tests/presentation/test_repository_config.py
git commit -m "ci: publish committed notebook presentations"
```

---

### Task 11: User documentation and discoverable README workflow

**Files:**
- Create: `docs/presentation-export.md`
- Modify: `README.md`
- Create: `tests/presentation/test_documentation.py`

**Interfaces:**
- Consumes: final CLI and output behavior.
- Produces: complete standalone operator documentation.

- [ ] **Step 1: Write failing documentation-presence tests**

Assert README links to `docs/presentation-export.md`, documented commands exactly match
CLI help, and the guide contains headings for Installation, Export, Validation,
`nbstripout`, Offline use, PDF, Plotly, Widgets and maps, Git LFS, Freshness,
Placeholders/WIP, Future notebooks, Troubleshooting, and GitHub Pages.

- [ ] **Step 2: Run documentation tests and confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_documentation.py -q`

Expected: FAIL because documentation is absent.

- [ ] **Step 3: Write the complete operator guide**

Document these exact workflows:

```bash
uv sync --extra presentation
uv run python scripts/export_notebooks.py --all
uv run python scripts/export_notebooks.py notebooks/02_U_Phase.ipynb
uv run python scripts/export_notebooks.py --all --strict
uv run python scripts/export_notebooks.py --check
```

Explain warning codes and strict blockers, the run-review-export-stage-commit order,
why source hashes survive `nbstripout`, why the complete folder must be copied, how
lazy Plotly works under `file://`, the OSM tile limitation, widget/Folium fallback,
browser PDF, manifest freshness, future notebook discovery, placeholder override,
missing-resource remediation, LFS requirements, and Pages manual dispatch.

State that Pages Source must remain `GitHub Actions` and that neither suggested GitHub
workflow template should be selected.

- [ ] **Step 4: Add a concise README section**

Add a short `## Notebook-Präsentationen` section after setup or documentation with one
install command, one all-export command, output path, and the guide link. Do not copy
the full guide into README.

- [ ] **Step 5: Run documentation tests and Markdown checks**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest tests/presentation/test_documentation.py -q
git diff --check -- README.md docs/presentation-export.md
```

Expected: PASS with no whitespace errors.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md docs/presentation-export.md tests/presentation/test_documentation.py
git commit -m "docs: explain notebook presentation exports"
```

---

### Task 12: Browser verification, real Q/U smoke export, full regression, and review

**Files:**
- Create: `tests/presentation/test_browser.py`
- Create: `tests/presentation/fixtures/gallery.ipynb`
- Generate: `reports/presentation/**`
- Modify only if defects are found: focused presentation implementation/test files.

**Interfaces:**
- Consumes: complete exporter.
- Produces: validated Q/U presentation artifacts and verification evidence.

- [ ] **Step 1: Create the representative gallery fixture**

Build a small saved notebook containing h1–h4 headings with duplicated umlaut titles,
collapsed code, executed outputless setup, text output, a 200-row/wide HTML table,
long log, SVG, PNG, small Plotly MIME, and synthetic warning metadata. The fixture has
stored outputs only and no code is executed to create it during tests.

- [ ] **Step 2: Add opt-in Playwright browser tests**

Mark tests `@pytest.mark.browser`. Open the exported gallery by `Path.as_uri()` and
test viewport sizes 1440×900, 1366×768, and 390×844. Assert:

- TOC is visible/drawer-accessible without overlapping main content;
- keyboard Tab reaches each global control with visible focus;
- code/output global and individual toggles update `open` and ARIA state;
- session state survives reload;
- table has horizontal overflow and expand action;
- Plotly container receives `.js-plotly-plot` after lazy load;
- no unexpected `request` uses `http://` or `https://`;
- `page.on("console")` and `page.on("pageerror")` collect no errors;
- reduced-motion emulation disables transitions;
- print media hides controls and exposes closed detail contents.

- [ ] **Step 3: Install the opt-in browser and run gallery tests**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra presentation --extra presentation-test
UV_CACHE_DIR=/tmp/uv-cache uv run playwright install chromium
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/presentation/test_browser.py -m browser -q
```

Expected: all browser tests PASS. Save desktop, laptop, mobile, and print screenshots
under `/tmp/unfallatlas-presentation-verification/`, not in the repository.

- [ ] **Step 4: Export only the currently finished real notebooks**

First confirm no training process is writing U, record working-tree status, and capture
the exact notebook bytes before export:

```bash
git status --short
sha256sum notebooks/01_Q_Phase.ipynb notebooks/02_U_Phase.ipynb \
  notebooks/03_A3_Phase.ipynb > /tmp/presentation-notebook-hashes.before
```

Then run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation python scripts/export_notebooks.py \
  notebooks/01_Q_Phase.ipynb notebooks/02_U_Phase.ipynb
```

Expected: exit 0; Q and U HTML files plus index/manifest exist; U reports large-output
and external-map warnings; A³ and C are not exported or modified.

- [ ] **Step 5: Measure and inspect real artifacts**

Run:

```bash
find reports/presentation -type f -printf '%s\t%p\n' | sort -n
rg -n "https?://|cdn|unpkg|jsdelivr" reports/presentation --glob '*.html' --glob '*.js'
sha256sum notebooks/01_Q_Phase.ipynb notebooks/02_U_Phase.ipynb \
  notebooks/03_A3_Phase.ipynb > /tmp/presentation-notebook-hashes.after
diff -u /tmp/presentation-notebook-hashes.before /tmp/presentation-notebook-hashes.after
```

Expected: notebook HTML files remain small relative to U's payload; Plotly data assets
contain the large bytes; no Plotly/MathJax CDN runtime; the hash comparison is empty,
proving the exporter did not alter Q, U, or A³ despite their pre-existing Git state.
Scientific hyperlinks may appear and must be distinguished from runtime URLs.

Open Q and U with Playwright. Load at least two small U Plotly charts and one largest
chart individually. Confirm initial page load does not parse every Plotly payload,
layout remains responsive, console is clean, and the map displays its offline warning.

- [ ] **Step 6: Verify normal, strict, stale, and batch behavior**

Run freshness and strict behavior against the real published selection, then use the
isolated CLI integration test for `--all` so no A³ presentation artifact is created:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation python scripts/export_notebooks.py --check
UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation python scripts/export_notebooks.py notebooks/02_U_Phase.ipynb --strict
UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest \
  tests/presentation/test_cli.py -k 'all or continues_after_failure' -q
```

Expected: `--check` reports Q/U fresh; strict U exits 1 for external OSM tiles; `--all`
fixture coverage proves deterministic discovery, continuation across failures, and
placeholder skipping. Separate discovery/status tests prove that the saved C skeleton
is a placeholder and unexecuted A³-style notebooks are WIP; neither is published as a
finished real presentation.

- [ ] **Step 7: Run complete automated verification**

Invoke the `superpowers:verification-before-completion` skill, then run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/uv-cache uv run --extra presentation pytest
UV_CACHE_DIR=/tmp/uv-cache uv run pre-commit run --all-files
```

Expected: all commands exit 0. Inspect `git status --short` and account for every path;
no user-owned notebook/model change is overwritten or staged accidentally.

- [ ] **Step 8: Commit generated Q/U presentation artifacts intentionally**

Verify LFS attributes before staging:

```bash
git check-attr filter -- reports/presentation/assets/vendor/* reports/presentation/assets/notebooks/**/*
git add reports/presentation
git lfs ls-files reports/presentation
git commit -m "docs(presentation): publish Q and U notebook snapshots"
```

Expected: large vendor/notebook assets are LFS pointers; index/manifest/HTML/UI remain
present; commit hooks pass. Do not stage A³ notebook/model changes with this commit.

- [ ] **Step 9: Request independent code review and resolve findings**

Invoke `superpowers:requesting-code-review` with the approved spec, this plan, the
implementation commit range, working-tree caveats, and verification results. For each
actionable finding, add or adjust a failing test first, implement the minimal fix, run
the narrow and full relevant tests, and commit with a scoped `fix(presentation): ...`
message.

- [ ] **Step 10: Re-run final verification after review**

Invoke `superpowers:verification-before-completion` again and repeat Ruff, full pytest,
pre-commit, browser checks, Q/U freshness, URL scan, LFS inspection, and `git status`.
Only after fresh successful output may completion be reported.
