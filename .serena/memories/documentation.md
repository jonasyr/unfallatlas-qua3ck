# Documentation Layout

`docs/` is organized by purpose, not flat:

- `docs/AI TOOL DISCLOSURE.md`, `docs/GLOSSARY.md` — hard requirements, stay at `docs/` top level. Never move.
- `docs/prompts/` — verbatim AI prompt transcripts per QUA³CK phase (`01_prompts_phase_q.md`, `02_prompts_phase_u.md`), referenced by AI TOOL DISCLOSURE.md. Text inside the fenced/quoted prompt blocks is a historical record — never edit it, even if it references paths that have since moved. Only the surrounding metadata (e.g. `**Kontext**:` bullets) may be corrected.
- `docs/course-material/` — lecture notes used as AI context (Einheit 1/2, "Data Analytics und Big Data", "Master Data Analysis with ChatGPT").
- `docs/dataset/` — Unfallatlas dataset description (`DSB_Unfallatlas.md`/`.pdf`); source of truth for coded categorical labels used in notebook label dictionaries (see `mem:conventions` for the U-Phase label-dict convention).
- `docs/project/` — repo/process docs (`ConventionalCommitsGuide.md`, `PROJEKTPLAN_SETUP.md`, `Technical_Review_Next_Steps.md` — German technical review of the 3-class gate-miss driving the binary KSI reframe).
- `docs/osm-feature-retrospective.md` — standalone record of the OSM road-context feature build (U-phase) and its measured effect on the A³ retrain; not part of the QUA³CK notebook chain.
- `docs/presentation-export.md` — German user guide for the notebook→HTML presentation exporter (`scripts/export_notebooks.py`, `src/unfallatlas/presentation/`): install, export/strict/check flags, validation finding codes, nbstripout ordering, Git LFS handling, GitHub Pages deploy.
- `docs/superpowers/specs/` — design specs backing implementation plans in `docs/superpowers/plans/`; both are now tracked/committed (the `docs/superpowers/` `.gitignore` entry is commented out — verify with `git check-ignore` before assuming otherwise).

When moving or adding a doc file: grep the whole repo (`AGENTS.md`, `.serena/memories/`, `docs/prompts/` non-fenced metadata) for the old filename before committing the move — references are not auto-tracked.
