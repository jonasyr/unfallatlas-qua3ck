# Docs Folder Tidy-Up + Serena Memory Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the flat `docs/` folder into topic subfolders (without breaking any link in the repo), then bring Serena's project memories up to date with the new structure.

**Architecture:** Pure file moves (`git mv`) into three new subfolders (`docs/course-material/`, `docs/dataset/`, `docs/project/`), keeping `docs/GLOSSARY.md`, `docs/AI TOOL DISCLOSURE.md`, and `docs/prompts/` at the existing top level (hard requirements / already tidy). Every reference to a moved file is grep-verified and updated, except references that live **inside verbatim quoted AI-prompt text** in `docs/prompts/01_prompts_phase_q.md` and `docs/prompts/02_prompts_phase_u.md` — those are historical transcripts of what was literally sent to a model and must not be rewritten (confirmed with user). Serena memory `conventions.md` gets its stale path fixed, and a new `documentation.md` memory is added describing the `docs/` layout so future agents don't have to rediscover it.

**Tech Stack:** Git (`git mv` for history-preserving moves), grep for reference discovery/verification, Serena MCP memory tools (`write_memory`, `edit_memory`) for memory maintenance.

## Global Constraints

- `docs/GLOSSARY.md` and `docs/AI TOOL DISCLOSURE.md` MUST stay at `docs/` top level — they are the two hard-required deliverables and are linked from the README/AGENTS.md by bare filename convention.
- `docs/prompts/` MUST stay at `docs/prompts/` — already correctly organized, and AGENTS.md explicitly documents this path (`docs/prompts/`, corrected from an earlier `docs/docs/prompts/` typo per git-insights history).
- Do NOT edit any text inside the fenced/quoted prompt blocks (between ` ```markdown `/` ```` md ` fences) in `docs/prompts/01_prompts_phase_q.md` or `docs/prompts/02_prompts_phase_u.md`. Only the surrounding metadata (e.g. the `**Kontext**:` bullet list, which sits *outside* the fence) may be updated.
- Use `git mv`, not `mv` + `git add`, so file history is preserved.
- After moves, a repo-wide grep for every moved filename (outside `docs/prompts/`'s fenced blocks) must return zero stale hits.

---

## File Structure

Before:
```
docs/
├── AI TOOL DISCLOSURE.md
├── ConventionalCommitsGuide.md
├── DSB_Unfallatlas.md
├── DSB_Unfallatlas.pdf
├── Data Analytics und Big Data.md
├── Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte.md
├── Einheit 2 – Understanding the Data - Datenexploration und Datenvorbereitung.md
├── GLOSSARY.md
├── Master Data Analysis with ChatGPT.md
├── PROJEKTPLAN_SETUP.md
└── prompts/
    ├── 01_prompts_phase_q.md
    └── 02_prompts_phase_u.md
```

After:
```
docs/
├── AI TOOL DISCLOSURE.md                 # unchanged location (hard requirement)
├── GLOSSARY.md                           # unchanged location (hard requirement)
├── prompts/                              # unchanged location
│   ├── 01_prompts_phase_q.md
│   └── 02_prompts_phase_u.md
├── course-material/                      # NEW — lecture notes used as AI context
│   ├── Data Analytics und Big Data.md
│   ├── Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte.md
│   ├── Einheit 2 – Understanding the Data - Datenexploration und Datenvorbereitung.md
│   └── Master Data Analysis with ChatGPT.md
├── dataset/                               # NEW — dataset documentation (for citing)
│   ├── DSB_Unfallatlas.md
│   └── DSB_Unfallatlas.pdf
└── project/                               # NEW — repo/process documentation
    ├── ConventionalCommitsGuide.md
    └── PROJEKTPLAN_SETUP.md
```

Known reference sites (discovered via repo-wide grep, see Task 1 step 1):
- `AGENTS.md:69` — tree diagram comment `docs/  # Glossary and documentation`
- `AGENTS.md:170` — git-insights bullet: `` sourced from `docs/DSB_Unfallatlas.md` ``
- `.serena/memories/conventions.md:27` — `` Files in `docs/ConventionalCommitsGuide.md` for reference ``
- `docs/prompts/01_prompts_phase_q.md:9-10` — `**Kontext**:` bullets (outside fence) pointing at `` `docs/Master Data Analysis with ChatGPT.md` `` and `` `docs/Data Analytics und Big Data.md` ``
- `docs/prompts/02_prompts_phase_u.md:884,952` — **inside** fenced prompt blocks referencing `` `docs/DSB_Unfallatlas.md` `` — **do not touch** (verbatim historical prompt text)
- `docs/prompts/02_prompts_phase_u.md:733` — **inside** fenced prompt block, a hypothetical path `docs/quaack/u_phase.md` suggested by the model — not a reference to a real moved file, **do not touch**

---

## Task 1: Move files into topic subfolders

**Files:**
- Create dirs: `docs/course-material/`, `docs/dataset/`, `docs/project/`
- Move (via `git mv`):
  - `docs/Data Analytics und Big Data.md` → `docs/course-material/Data Analytics und Big Data.md`
  - `docs/Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte.md` → `docs/course-material/Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte.md`
  - `docs/Einheit 2 – Understanding the Data - Datenexploration und Datenvorbereitung.md` → `docs/course-material/Einheit 2 – Understanding the Data - Datenexploration und Datenvorbereitung.md`
  - `docs/Master Data Analysis with ChatGPT.md` → `docs/course-material/Master Data Analysis with ChatGPT.md`
  - `docs/DSB_Unfallatlas.md` → `docs/dataset/DSB_Unfallatlas.md`
  - `docs/DSB_Unfallatlas.pdf` → `docs/dataset/DSB_Unfallatlas.pdf`
  - `docs/ConventionalCommitsGuide.md` → `docs/project/ConventionalCommitsGuide.md`
  - `docs/PROJEKTPLAN_SETUP.md` → `docs/project/PROJEKTPLAN_SETUP.md`

- [ ] **Step 1: Confirm the full set of reference sites before moving anything**

Run (from repo root):
```bash
cd /home/jonas/Documents/Code/unfallatlas-qua3ck
grep -rn "ConventionalCommitsGuide\|PROJEKTPLAN_SETUP\|DSB_Unfallatlas\|Master Data Analysis with ChatGPT\|Data Analytics und Big Data\|Einheit 1 –\|Einheit 2 –" \
  --exclude-dir=.git --exclude-dir=.venv --exclude-dir=.serena/cache .
```
Expected: matches only in `AGENTS.md`, `.serena/memories/conventions.md`, `docs/prompts/01_prompts_phase_q.md`, `docs/prompts/02_prompts_phase_u.md`, and the docs files themselves (self/sibling references via Obsidian `[[wikilinks]]`, which are filename-based and need no path update). If new/unexpected files show up here, note them — they must be covered by Task 2/3 too.

- [ ] **Step 2: Create the new subfolders and move the files with `git mv`**

```bash
cd /home/jonas/Documents/Code/unfallatlas-qua3ck
mkdir -p "docs/course-material" "docs/dataset" "docs/project"
git mv "docs/Data Analytics und Big Data.md" "docs/course-material/Data Analytics und Big Data.md"
git mv "docs/Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte.md" "docs/course-material/Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte.md"
git mv "docs/Einheit 2 – Understanding the Data - Datenexploration und Datenvorbereitung.md" "docs/course-material/Einheit 2 – Understanding the Data - Datenexploration und Datenvorbereitung.md"
git mv "docs/Master Data Analysis with ChatGPT.md" "docs/course-material/Master Data Analysis with ChatGPT.md"
git mv "docs/DSB_Unfallatlas.md" "docs/dataset/DSB_Unfallatlas.md"
git mv "docs/DSB_Unfallatlas.pdf" "docs/dataset/DSB_Unfallatlas.pdf"
git mv "docs/ConventionalCommitsGuide.md" "docs/project/ConventionalCommitsGuide.md"
git mv "docs/PROJEKTPLAN_SETUP.md" "docs/project/PROJEKTPLAN_SETUP.md"
```

- [ ] **Step 3: Verify the moves landed correctly and history is preserved**

```bash
git status --short
git log --oneline --follow -1 -- "docs/dataset/DSB_Unfallatlas.md"
```
Expected: `git status --short` shows 8 `R  ` (renamed) entries, no `D`/`??` pairs; `git log --follow` returns at least one prior commit (history preserved through the rename).

- [ ] **Step 4: Confirm `docs/GLOSSARY.md`, `docs/AI TOOL DISCLOSURE.md`, and `docs/prompts/` were not touched**

```bash
ls "docs/AI TOOL DISCLOSURE.md" docs/GLOSSARY.md docs/prompts/01_prompts_phase_q.md docs/prompts/02_prompts_phase_u.md
```
Expected: all four paths exist, unchanged.

---

## Task 2: Update references in AGENTS.md

**Files:**
- Modify: `AGENTS.md:69` (tree diagram), `AGENTS.md:170` (git-insights bullet)

**Interfaces:**
- Consumes: new paths from Task 1 (`docs/dataset/DSB_Unfallatlas.md`, `docs/course-material/`, `docs/project/`)

- [ ] **Step 1: Update the directory tree comment**

Find this line in `AGENTS.md` (inside the `<!-- AUTO-MANAGED: architecture -->` block):
```
├── docs/                   # Glossary and documentation
│   └── prompts/            # AI prompts used per QUA³CK phase (01_..., 02_..., referenced by AI_TOOL_DISCLOSURE.md)
```
Replace with:
```
├── docs/                   # Disclosure + glossary (hard requirements) and supporting docs
│   ├── prompts/            # AI prompts used per QUA³CK phase (01_..., 02_..., referenced by AI TOOL DISCLOSURE.md)
│   ├── course-material/    # Lecture notes used as AI context (Einheit 1/2, Data Analytics und Big Data, ChatGPT best-practice notes)
│   ├── dataset/            # Unfallatlas dataset description (DSB_Unfallatlas.md/.pdf), used for citing + coded-label lookups
│   └── project/            # Repo/process docs (ConventionalCommitsGuide.md, PROJEKTPLAN_SETUP.md)
```

- [ ] **Step 2: Fix the stale `docs/DSB_Unfallatlas.md` path in git-insights**

Find this line (inside the `<!-- AUTO-MANAGED: git-insights -->` block):
```
- U-Phase plotting conventions documented in `docs/prompts/02_prompts_phase_u.md`: human-readable label dicts (`FEATURE_LABELS`, `UKATGEORIE_LABELS`, `COL_CODE_LABELS`, etc.) + helpers (`feature_label()`, `severity_label()`, `apply_code_labels()`) sourced from `docs/DSB_Unfallatlas.md`; consistent `sns.set_theme(style="whitegrid", palette="colorblind")` styling
```
Replace `docs/DSB_Unfallatlas.md` with `docs/dataset/DSB_Unfallatlas.md` (rest of the line unchanged).

- [ ] **Step 3: Add a one-line git-insights note about the docs reorg**

Append a new bullet at the end of the `<!-- AUTO-MANAGED: git-insights -->` block:
```
- `docs/` reorganized into `prompts/`, `course-material/`, `dataset/`, `project/`; `GLOSSARY.md` and `AI TOOL DISCLOSURE.md` stay at `docs/` top level (hard requirements)
```

- [ ] **Step 4: Verify**

```bash
grep -n "docs/DSB_Unfallatlas.md\b" AGENTS.md   # expect: no output (old bare path gone)
grep -n "docs/dataset/DSB_Unfallatlas.md\|docs/course-material/\|docs/project/" AGENTS.md   # expect: 3+ matches
```

---

## Task 3: Update the non-quoted metadata reference in docs/prompts/01_prompts_phase_q.md

**Files:**
- Modify: `docs/prompts/01_prompts_phase_q.md:9-10`

**Interfaces:**
- Consumes: new path `docs/course-material/` from Task 1

- [ ] **Step 1: Confirm the two lines are outside any fence**

```bash
sed -n '1,15p' docs/prompts/01_prompts_phase_q.md
```
Expected: lines 9-10 (the `**Kontext**:` bullets) appear *before* the first ` ```markdown ` fence opens at line 12 — confirming they are plain metadata, not quoted prompt text.

- [ ] **Step 2: Update the two bullet paths**

Find:
```
**Kontext**:

- `docs/Master Data Analysis with ChatGPT.md`
- `docs/Data Analytics und Big Data.md`
```
Replace with:
```
**Kontext**:

- `docs/course-material/Master Data Analysis with ChatGPT.md`
- `docs/course-material/Data Analytics und Big Data.md`
```

- [ ] **Step 3: Verify nothing inside the fenced prompt blocks changed**

```bash
git diff docs/prompts/01_prompts_phase_q.md
```
Expected: exactly 2 changed lines (the two `Kontext` bullets), both above the first ` ```markdown ` fence; no other lines touched.

---

## Task 4: Fix the stale path in Serena memory `conventions.md`

**Files:**
- Modify (via Serena `edit_memory` tool, not a raw file edit): memory `conventions`

- [ ] **Step 1: Read the current memory to confirm exact text**

Use Serena's `read_memory` tool with `memory_name="conventions"`. Confirm the line reads:
```
- Files in `docs/ConventionalCommitsGuide.md` for reference
```

- [ ] **Step 2: Apply the fix**

Use Serena's `edit_memory` tool:
- `memory_name`: `"conventions"`
- `mode`: `"literal"`
- `needle`: `` "Files in `docs/ConventionalCommitsGuide.md` for reference" ``
- `repl`: `` "Files in `docs/project/ConventionalCommitsGuide.md` for reference" ``
- `allow_multiple_occurrences`: `false`

- [ ] **Step 3: Verify**

Use Serena's `read_memory` tool with `memory_name="conventions"` again; confirm the line now reads `` Files in `docs/project/ConventionalCommitsGuide.md` for reference ``.

---

## Task 5: Add a Serena memory documenting the new docs/ layout

**Files:**
- Create (via Serena `write_memory` tool, not a raw file write): memory `documentation`
- Modify (via Serena `edit_memory` tool): memory `core` (add a reference line)

This is a genuinely new, stable, non-obvious project convention (per the memory-maintenance threshold already encoded in `mem:memory_maintenance`: "stable, non-obvious project conventions that avoid complex rediscovery"), so it earns its own memory rather than bloating `core`.

- [ ] **Step 1: Write the new memory**

Use Serena's `write_memory` tool with `memory_name="documentation"` and this content:
```markdown
# Documentation Layout

`docs/` is organized by purpose, not flat:

- `docs/AI TOOL DISCLOSURE.md`, `docs/GLOSSARY.md` — hard requirements, stay at `docs/` top level. Never move.
- `docs/prompts/` — verbatim AI prompt transcripts per QUA³CK phase (`01_prompts_phase_q.md`, `02_prompts_phase_u.md`), referenced by AI TOOL DISCLOSURE.md. Text inside the fenced/quoted prompt blocks is a historical record — never edit it, even if it references paths that have since moved. Only the surrounding metadata (e.g. `**Kontext**:` bullets) may be corrected.
- `docs/course-material/` — lecture notes used as AI context (Einheit 1/2, "Data Analytics und Big Data", "Master Data Analysis with ChatGPT").
- `docs/dataset/` — Unfallatlas dataset description (`DSB_Unfallatlas.md`/`.pdf`); source of truth for coded categorical labels used in notebook label dictionaries (see `mem:conventions` for the U-Phase label-dict convention).
- `docs/project/` — repo/process docs (`ConventionalCommitsGuide.md`, `PROJEKTPLAN_SETUP.md`).

When moving or adding a doc file: grep the whole repo (`AGENTS.md`, `.serena/memories/`, `docs/prompts/` non-fenced metadata) for the old filename before committing the move — references are not auto-tracked.
```

- [ ] **Step 2: Link it from `core`**

Use Serena's `edit_memory` tool on `memory_name="core"`:
- `mode`: `"literal"`
- `needle`: `` "See `mem:tech_stack` for dependencies, `mem:conventions` for code style, `mem:suggested_commands` for CLI usage, `mem:task_completion` for done checklist." ``
- `repl`: `` "See `mem:tech_stack` for dependencies, `mem:conventions` for code style, `mem:suggested_commands` for CLI usage, `mem:task_completion` for done checklist, `mem:documentation` for the `docs/` folder layout." ``
- `allow_multiple_occurrences`: `false`

- [ ] **Step 3: Verify**

Use Serena's `list_memories` tool; confirm `documentation` now appears in the list. Use `read_memory` on `core`; confirm the new `mem:documentation` reference is present.

---

## Task 6: Repo-wide verification pass

**Files:** none (read-only verification)

- [ ] **Step 1: Confirm no stale references remain outside the protected fenced blocks**

```bash
cd /home/jonas/Documents/Code/unfallatlas-qua3ck
grep -rn "docs/ConventionalCommitsGuide\.md\|docs/PROJEKTPLAN_SETUP\.md\|docs/DSB_Unfallatlas\.\(md\|pdf\)\|docs/Master Data Analysis with ChatGPT\.md\|docs/Data Analytics und Big Data\.md\|docs/Einheit 1 –\|docs/Einheit 2 –" \
  --exclude-dir=.git --exclude-dir=.venv .
```
Expected: only hits remaining are inside the fenced prompt blocks of `docs/prompts/02_prompts_phase_u.md` (lines ~884, ~952 — `docs/DSB_Unfallatlas.md` inside a ` ```markdown `/` ```` md ` block) — these are the intentionally-preserved historical transcript references confirmed with the user in Task constraints. No hits in `AGENTS.md`, `.serena/memories/`, `README.md`, or the non-fenced part of `docs/prompts/01_prompts_phase_q.md`.

- [ ] **Step 2: Confirm the new paths resolve**

```bash
test -f "docs/course-material/Data Analytics und Big Data.md" && \
test -f "docs/course-material/Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte.md" && \
test -f "docs/course-material/Einheit 2 – Understanding the Data - Datenexploration und Datenvorbereitung.md" && \
test -f "docs/course-material/Master Data Analysis with ChatGPT.md" && \
test -f "docs/dataset/DSB_Unfallatlas.md" && \
test -f "docs/dataset/DSB_Unfallatlas.pdf" && \
test -f "docs/project/ConventionalCommitsGuide.md" && \
test -f "docs/project/PROJEKTPLAN_SETUP.md" && \
echo OK
```
Expected: prints `OK`.

- [ ] **Step 3: Run pre-commit on the touched files (docs are markdown, but the repo's pre-commit runs repo-wide checks like large-file/private-key detection)**

```bash
pre-commit run --files AGENTS.md ".serena/memories/conventions.md" \
  "docs/prompts/01_prompts_phase_q.md" \
  "docs/course-material/Data Analytics und Big Data.md" \
  "docs/course-material/Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte.md" \
  "docs/course-material/Einheit 2 – Understanding the Data - Datenexploration und Datenvorbereitung.md" \
  "docs/course-material/Master Data Analysis with ChatGPT.md" \
  "docs/dataset/DSB_Unfallatlas.md" "docs/dataset/DSB_Unfallatlas.pdf" \
  "docs/project/ConventionalCommitsGuide.md" "docs/project/PROJEKTPLAN_SETUP.md"
```
Expected: hooks pass (or are skipped as not-applicable for markdown/pdf); no failures.

- [ ] **Step 4: Stage and review the full diff before handing back to the user**

```bash
git add -A
git status --short
```
Expected: 8 renames (`R`), modifications to `AGENTS.md` and `docs/prompts/01_prompts_phase_q.md`. Do **not** commit — leave staged for the user to review and commit themselves.

---

## Self-Review Notes

- **Spec coverage:** (1) tidy docs folder ✔ Task 1; (2) keep GLOSSARY.md + AI TOOL DISCLOSURE.md as hard requirements at top level ✔ Task 1 constraint, Task 6 verification; (3) all references to moved files updated everywhere ✔ Tasks 2/3/6, with the one explicit, user-confirmed exception (verbatim quoted historical prompts); (4) update Serena memories ✔ Tasks 4/5; (5) use Serena where it makes sense ✔ Tasks 4/5 use Serena memory tools exclusively for memory edits (file moves themselves are plain `git mv`, which is correctly outside Serena's code-symbol scope since these are not source-code symbols).
- **Placeholder scan:** no TBD/TODO; every step has the literal command or text to use.
- **Type/name consistency:** memory name `documentation` used consistently across Task 5 steps 1-3; path names consistent across Tasks 1, 2, 3, 6.
