# Notebook Presentation Export — Design Specification

**Date:** 2026-07-14  
**Status:** Approved  
**Repository:** `unfallatlas-qua3ck`

## 1. Purpose

Build a professional, scalable export workflow that converts saved Jupyter notebook
state into presentation-ready HTML snapshots. The exporter must preserve existing
outputs without executing notebook code. A presentation must open directly in a
browser on a lower-powered laptop without Python, Jupyter, the project dataset, or
trained models.

The portable unit is the complete `reports/presentation/` directory. Each selected
notebook has one HTML document, while large rich outputs and shared browser libraries
are stored as local assets. This preserves Plotly interactivity without forcing the
browser to parse every chart when the document first opens.

## 2. Verified Repository State

The design is based on the local repository rather than the initial task assumptions.

- `notebooks/*.ipynb` are the source of truth. Paired `.py` files are generated
  Jupytext mirrors and must never be edited directly.
- Q is complete and intentionally contains 16 Markdown cells and no code cells.
- U is complete and was freshly executed during design work:
  - 55 of 55 code cells have execution counts 1 through 55;
  - the counts are unique and monotonic;
  - 52 code cells have outputs;
  - three executed setup cells legitimately have no output;
  - no Error outputs are stored;
  - 25 outputs use `application/vnd.plotly.v1+json`;
  - Plotly payloads total about 216.7 MB;
  - six individual Plotly payloads range from about 22.3 MB to 62.0 MB;
  - the executed notebook is about 263.2 MB.
- A³ is work in progress and was training models while this design was written. It
  must not be changed or presented as complete.
- C is a 499-byte, one-cell TODO placeholder.
- K has not started; `app/streamlit_app.py` is empty.
- `reports/presentation/` does not exist yet.
- `reports/figures/` is ignored except for `.gitkeep`. It contains 26 generated
  Plotly HTML files ranging from about 8 KB to 68.4 MB. They depend on the Plotly CDN.
- The U notebook includes one LaTeX expression and one Plotly OpenStreetMap chart.
  The map's basemap requires online tiles.
- Current stored outputs do not include ipywidgets or Folium.
- DataFrame outputs include the VS Code Data Wrangler MIME type alongside usable
  `text/html` and `text/plain` fallbacks.
- `nbformat` is a direct dependency. `nbconvert` 7.17.1 is currently only transitive
  through the development Jupyter dependency.
- Pre-commit strips notebook outputs with `nbstripout` and rejects added files over
  5 MB. Git LFS is already used for the dataset and fitted model artifacts.
- `reports/presentation/` is not ignored.
- GitHub Pages is configured to use GitHub Actions, but no Pages workflow or site is
  deployed yet.
- The working tree contains user-owned changes to U and A³ plus an untracked SVM
  plan. Export implementation must preserve them.

The standard nbconvert templates cannot render the repository's saved Plotly MIME
bundles: a measured conversion emitted 25 unsupported-MIME warnings and omitted all
25 charts. The `basic` template without those charts was about 322 KB; the `lab`
template was about 629 KB. Therefore explicit Plotly MIME support is mandatory.

## 3. Chosen Architecture

Use a modular presentation package, a thin script entry point, a custom nbconvert
template, local browser assets, and content-addressed notebook-output assets.

```text
scripts/export_notebooks.py
src/unfallatlas/presentation/
├── __init__.py
├── discovery.py
├── validation.py
├── metadata.py
├── rendering.py
├── manifest.py
├── models.py
└── templates/unfallatlas/
    ├── conf.json
    ├── index.html.j2
    ├── presentation.css
    └── presentation.js
tests/
├── fixtures/notebooks/
└── test_presentation_*.py
```

Responsibilities are separated as follows:

- `discovery.py`: input selection, deterministic ordering, and notebook status.
- `validation.py`: notebook, execution, MIME, resource, offline, and size findings.
- `metadata.py`: timezone-aware export time, Git metadata, and reproducibility hashes.
- `rendering.py`: nbconvert configuration, template resources, rich-output assets,
  and atomic filesystem publication.
- `manifest.py`: manifest merge, freshness checks, and the generated index page.
- `models.py`: typed immutable or dataclass-based results shared across modules.
- `scripts/export_notebooks.py`: only argument parsing, orchestration, summaries, and
  exit codes.

The package must not import or use `ExecutePreprocessor`, start a kernel, invoke
Jupyter execution, or modify the source NotebookNode on disk. Rendering works on an
in-memory copy.

## 4. Output Layout

```text
reports/presentation/
├── index.html
├── manifest.json
├── notebooks/
│   ├── 01_Q_Phase.html
│   └── 02_U_Phase.html
└── assets/
    ├── ui/
    │   ├── presentation.css
    │   └── presentation.js
    ├── vendor/
    │   ├── plotly-<version>.min.js
    │   └── mathjax-<version>-tex-svg-full.js
    └── notebooks/
        └── 02_U_Phase/
            ├── plotly-<cell>-<digest>.js
            └── output-<cell>-<digest>.<ext>
```

The complete directory is copied to another computer. Copying only a notebook HTML
file is not a supported offline workflow because shared and large assets are local
siblings.

Notebook output names derive from the input stem. Assets include a cell identifier
and content digest. New assets are written before the HTML that references them.
HTML, manifest, and index replacements use a temporary file in the target directory,
flush and close it, then call `os.replace`.

The exporter never deletes presentation files. Removed or renamed sources become
orphaned manifest entries. Automatic pruning is outside the first implementation.

## 5. Discovery and Status

`--all` recursively discovers only `.ipynb` files under `notebooks/`. It ignores
Jupytext `.py` mirrors and sorts paths by normalized relative POSIX path.

Status precedence is:

1. optional explicit `metadata.presentation.status` value (`ready`, `wip`, or
   `placeholder`);
2. conservative placeholder recognition;
3. execution-state inference;
4. ready.

A notebook is inferred to be a placeholder only when all conditions hold:

- it has at most two non-empty content cells;
- it has no non-empty code cells;
- its initial content contains an explicit case-insensitive `TODO`, `TBD`, or
  `Platzhalter` marker.

This recognizes C without classifying the Markdown-only Q notebook as incomplete.
TODO comments inside a developed notebook do not make it a placeholder.

A notebook with non-empty unexecuted code, stored errors, or explicit `wip` metadata
is incomplete. Normal mode may export it, but the HTML and index identify it as WIP.
Strict mode rejects it. Placeholders are skipped by `--all` unless
`--include-placeholders` is supplied.

The first committed presentation set contains only finished Q and U. A³ is not
published until its current work is complete, and C/K are not presented as finished.

## 6. Command-Line Interface

Supported commands include:

```bash
uv run python scripts/export_notebooks.py --all
uv run python scripts/export_notebooks.py notebooks/02_U_Phase.ipynb
uv run python scripts/export_notebooks.py --all --strict
uv run python scripts/export_notebooks.py --all --open
uv run python scripts/export_notebooks.py --all --include-placeholders
uv run python scripts/export_notebooks.py --check
```

Options:

- positional notebook paths for explicit selection;
- `--all`, mutually exclusive with positional paths;
- `--output-dir`, defaulting to `reports/presentation/`;
- `--strict`;
- `--include-placeholders`;
- `--open`;
- `--check`, which compares source-content hashes with the manifest without export.

Explicit inputs must resolve to existing `.ipynb` files under `notebooks/`. A `.py`
mirror, path outside `notebooks/`, or placeholder without the include flag is a
selection error.

`--open` uses `webbrowser` and `Path.as_uri()`. For one notebook it opens that
document; for a batch it opens `index.html`.

Exit codes:

- `0`: every requested operation succeeded; warnings are permitted;
- `1`: at least one notebook failed, strict validation failed, or `--check` found a
  missing/stale export;
- `2`: invalid arguments or selection.

A batch continues after per-notebook failures. The final summary lists source,
status, destination, size, warning counts, and failure reason.

## 7. Validation Model

Every finding has a stable code, severity, message, optional cell index, and optional
remediation. Severities are `INFO`, `WARNING`, and `ERROR`.

Validation distinguishes:

- an empty or Markdown-only notebook;
- a non-empty code cell with no execution count;
- an executed code cell with no output;
- a cell with visible output;
- a cell with Error output.

Whitespace-only code is ignored. Executed outputless cells are informational and
never fail strict mode.

Always-fatal conditions include unreadable/invalid notebook JSON, unsafe input paths,
rendering failure, and atomic-publication failure.

Strict blockers include:

- non-empty unexecuted code;
- stored Error outputs;
- duplicate or non-monotonic execution counts;
- explicit or inferred WIP status;
- missing local runtime resources;
- path traversal, symlink escape, or resolved paths outside the repository;
- MIME output with no supported fallback;
- widget MIME without required stored state;
- runtime external scripts, styles, images, iframes, map tiles, or equivalent
  resources that prevent the promised offline result.

Normal mode exports strict findings as visible warnings whenever safe rendering is
possible. The current U notebook therefore exports normally with an offline warning
for its OpenStreetMap tiles, but fails `--strict` until that online dependency is
removed or replaced.

Normal scientific hyperlinks are not offline runtime dependencies. Resource checks
target `src`, executable `href`, CSS URLs, iframe sources, Plotly map layers, and
equivalent browser-loaded resources.

Size findings:

- any MIME payload over 5 MiB receives a large-output warning and is externalized;
- total output payload over 100 MiB receives a high-memory/performance warning;
- every generated file and the complete presentation directory are measured after
  publication.

## 8. HTML Structure and Accessibility

Each document contains semantic header, metadata, navigation, main content, and
footer regions. The design is restrained, light, projector-friendly, responsive,
and based on system fonts. It uses no decorative animation.

The header displays:

- title and status;
- source notebook;
- export timestamp in ISO 8601 and readable local form;
- full/short Git commit and branch where available;
- a conspicuous dirty-tree marker;
- Markdown and code cell counts;
- warning count and execution completeness.

Markdown is always visible. Code defaults closed and outputs default open. Each code
and output section uses native `<details>`/`<summary>`, so individual toggling works
without JavaScript. Global controls show/hide all code and all outputs. Their state is
stored per snapshot in `sessionStorage`.

Controls use native buttons, visible focus states, meaningful labels,
`aria-controls`, and `aria-expanded`. The document remains readable when custom
JavaScript fails. Plotly placeholders retain a title, size, and failure message if a
chart cannot initialize.

## 9. Table of Contents

The exporter derives the table of contents from Markdown `h1` through `h6` headings.
It creates deterministic unique Unicode IDs, preserving umlauts and suffixing
duplicates. Deep links are present in static HTML.

On wide screens the hierarchy is sticky beside the content. On narrow screens it is
a non-overlapping drawer. `IntersectionObserver` enhances the active-section state;
basic links remain functional without JavaScript. A keyboard-accessible back-to-top
control is included.

## 10. Rich Output Strategy

### 10.1 Plotly

The custom template explicitly supports `application/vnd.plotly.v1+json`.

- The installed Plotly package supplies one versioned local `plotly.min.js` copy.
- Each figure specification is serialized once to a content-addressed JavaScript
  asset.
- Assets register their figure data under a stable chart key. Dynamic local script
  loading is used instead of `fetch()`, so direct `file://` viewing works without a
  local server or browser CORS exceptions.
- A chart loads when its output is opened or approaches the viewport.
- `Plotly.newPlot` uses responsive configuration.
- The output document does not include external Plotly CDNs or duplicate libraries.

This preserves all stored U data while avoiding an initial parse of about 216.7 MB.
It does not reduce the total directory size.

Plotly charts that depend on external map tiles receive a visible offline limitation.
The exporter does not download or redistribute map tiles.

### 10.2 Images and SVG

PNG, JPEG, SVG, and other supported extracted outputs become content-addressed local
assets. Existing Markdown image references are resolved relative to the notebook,
validated against the repository root, and copied or embedded according to size.

SVG is treated as active-capable content and handled so that it cannot escape the
document's asset boundary. Missing or unsafe references produce findings rather than
silent broken images.

### 10.3 HTML, tables, text, and errors

Pandas and generic HTML tables are wrapped without assuming a single DataFrame DOM
shape. Tables scroll horizontally, have a bounded vertical viewport, use sticky
headers where possible, and can be expanded. Text, logs, and tracebacks have bounded
scrollable containers without destructive truncation.

The Data Wrangler MIME is ignored when the same output provides `text/html` or
`text/plain`. Active HTML/script output is isolated in a restricted sandbox iframe.

### 10.4 Mathematics

A pinned local MathJax TeX-to-SVG bundle and its license are included as a shared
offline asset. The export must not depend on the default nbconvert MathJax CDN.

### 10.5 Widgets, Folium, and iframes

Stored widgets are rendered only when their MIME data and widget state are complete.
Otherwise a clear fallback and warning are emitted. Strict mode rejects the case.

Folium and general iframes are best-effort. Their HTML is isolated; external Leaflet
libraries, tiles, or iframe URLs are reported. The exporter does not claim full
offline support when those resources remain external.

## 11. Printing and PDF

Print CSS hides navigation and interactive controls, uses an appropriate page width,
reduces unnecessary backgrounds, constrains figures to the page, improves heading
page breaks, and keeps code and tables readable.

Printed output includes code and outputs regardless of their screen `<details>`
state. A Print/PDF button first loads pending Plotly charts, then calls browser print.
`beforeprint` provides best-effort support for direct Ctrl/Cmd+P. Rendered Plotly SVG
is printed rather than requiring a separate static-image pipeline.

`prefers-reduced-motion` is respected.

## 12. Git and Freshness Metadata

Git calls use `subprocess.run` with argument lists, `shell=False`, captured text, and
controlled failure handling. Missing Git produces explicit `unknown` placeholders.

The manifest stores two hashes:

- snapshot hash: the complete saved notebook including outputs;
- source-content hash: a canonical notebook representation excluding outputs,
  execution counts, and other execution-only metadata.

Freshness comparisons use the source-content hash. Consequently, `nbstripout` does
not make a just-exported presentation immediately stale, while Markdown or code
changes do. The complete snapshot hash still identifies the exact exported output
state.

The Git commit describes repository HEAD. A dirty flag explicitly warns that the
snapshot may contain uncommitted changes, which is expected when exporting executed
outputs before pre-commit strips them.

`manifest.json` includes schema/exporter version, generated time, repository metadata,
and per-notebook status, hashes, counts, findings, asset list, and sizes.

## 13. Git LFS and Pre-commit

The 5-MB pre-commit threshold remains unchanged. `.gitattributes` receives narrowly
scoped LFS rules for generated presentation notebook-output assets and large vendor
assets. Small index, manifest, CSS, and UI JavaScript remain ordinary Git files.

The exporter reports files that should be LFS-managed and strict validation fails if
a generated oversized asset is outside the configured LFS scope. It does not bypass
or disable hooks.

The recommended commit sequence is:

1. execute and inspect the notebook on the powerful machine;
2. export selected presentations;
3. inspect warnings and the generated HTML;
4. stage the presentation, notebook, and intended source changes;
5. allow `nbstripout` to remove notebook outputs;
6. confirm the presentation assets remain staged and the source-content hash is
   still fresh.

## 14. Dependencies

Add an optional presentation extra:

```toml
[project.optional-dependencies]
presentation = [
    "beautifulsoup4>=4",
    "nbconvert>=7.17,<8",
]
```

`nbformat` and Plotly already exist as direct dependencies. nbconvert supplies Jinja2
and Pygments. No Node, Vite, Webpack, or Kaleido dependency is introduced.

Add an optional browser-test extra for Python Playwright. It is not installed or
needed for normal export. Document the separate browser installation command.

Update `uv.lock` with uv only. CI installs the presentation extra for unit and
integration tests.

## 15. GitHub Pages

Add `.github/workflows/pages.yml`. GitHub Pages remains optional for local use.

Triggers:

- pushes to `main` limited to `reports/presentation/**` and the Pages workflow;
- `workflow_dispatch`.

The workflow uses current official checkout/configure/upload/deploy Pages actions.
It checks out Git LFS content, validates that `index.html` and `manifest.json` exist,
checks total size, uploads only `reports/presentation/`, and deploys it.

It never installs Python, accesses the dataset, executes notebooks, or trains models.
Permissions are limited to `contents: read`, `pages: write`, and `id-token: write`.
The deployment uses the standard `github-pages` environment and concurrency control.

The published site must remain below GitHub Pages' 1-GB site limit. The exporter and
workflow warn before that boundary. U's expected roughly 220-MB presentation fits,
but future phases must be monitored.

The repository setting is already `Source: GitHub Actions`. The generic Jekyll and
Static HTML configuration buttons must not be used; the repository workflow is the
publishing source.

## 16. Testing Strategy

Implementation follows test-driven development. Fixtures are small and do not use
the computational project notebooks for routine tests.

Coverage includes:

- Markdown-only notebook;
- executed text output;
- unexecuted code;
- executed outputless code;
- Error output;
- DataFrame/generic HTML table;
- PNG and SVG;
- missing and traversal local resources;
- invalid notebook;
- placeholder and explicit WIP;
- normal and strict behavior;
- single and batch export;
- deterministic discovery and filenames;
- a newly added future notebook;
- Git metadata with/without Git and dirty state;
- atomic replacement failure preserving the old target;
- umlauts, duplicate headings, and stable anchors;
- global and individual code/output controls;
- basic accessibility attributes;
- Plotly MIME extraction, one shared runtime, lazy local references, and no CDN;
- output fallback selection;
- manifest merging, source hash stability after output stripping, staleness, and
  orphan status;
- CLI exit codes and continued batch processing;
- offline asset references and Pages artifact layout.

Assertions target semantic DOM fragments and metadata, not complete generated HTML
byte snapshots.

## 17. Visual Verification

Generate a representative gallery with nested headings, code, outputless setup,
text, long logs, a large table, image, Plotly chart, and warning metadata.

Use Python Playwright in an optional test environment to inspect:

- 1440×900 desktop;
- 1366×768 laptop;
- approximately 390×844 mobile;
- table scrolling and expansion;
- TOC and active section;
- individual/global toggles;
- keyboard focus;
- lazy Plotly rendering;
- offline network behavior;
- JavaScript console errors;
- print layout and browser PDF.

After gallery iteration, smoke-export Q and the real executed U, measure actual HTML
and asset sizes, and inspect small and large Plotly figures. Do not modify A³ while
its work is incomplete.

## 18. Documentation

Add a concise README section linking to `docs/presentation-export.md`.

The full guide covers installation, single/all export, warnings, strict mode, warning
codes, pre-commit order, nbstripout, output layout, copying the complete folder,
offline use, Plotly, mathematics, widgets, Folium/maps, PDF printing, Git metadata,
freshness, future notebooks, placeholders/WIP, missing resources, Git LFS, GitHub
Pages, and troubleshooting.

## 19. Verification and Review Gates

Before completion, run at minimum:

```bash
uv run ruff check .
uv run pytest
uv run pre-commit run --all-files
```

Also exercise CLI help, normal export, strict export, freshness checks, offline URL
scanning, visual browser checks, PDF output, and Git-diff inspection.

Use the Superpowers `verification-before-completion` workflow before claiming success.
Then run an independent `requesting-code-review` pass. Review findings re-enter the
test and verification loop.

## 20. Acceptance Mapping

The design satisfies the requested acceptance criteria by:

- never executing notebooks;
- using dynamic `.ipynb` discovery;
- treating Markdown-only Q as complete, U as executed, A³ as WIP, and C as a
  placeholder;
- preserving Markdown, code, stored outputs, and Plotly interactivity;
- using native and global code/output controls;
- generating stable navigation anchors;
- containing tables and long outputs;
- including Git/time/execution metadata and dirty status;
- exporting with warnings normally and failing defined issues in strict mode;
- providing print/PDF CSS;
- committing outputs under `reports/presentation/` with scoped LFS handling;
- validating with unit, integration, browser, real-notebook, lint, pre-commit, and
  review gates;
- publishing committed snapshots through an independent Pages workflow without any
  notebook execution.
