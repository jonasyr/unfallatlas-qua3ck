# Documentation Layout

`docs/` is organized by purpose, not flat:

- `docs/AI TOOL DISCLOSURE.md`, `docs/GLOSSARY.md` — hard requirements, stay at `docs/` top level. Never move.
- `docs/prompts/` — verbatim AI prompt transcripts per QUA³CK phase (`01_prompts_phase_q.md`, `02_prompts_phase_u.md`), referenced by AI TOOL DISCLOSURE.md. Text inside the fenced/quoted prompt blocks is a historical record — never edit it, even if it references paths that have since moved. Only the surrounding metadata (e.g. `**Kontext**:` bullets) may be corrected.
- `docs/course-material/` — lecture notes used as AI context (Einheit 1/2, "Data Analytics und Big Data", "Master Data Analysis with ChatGPT").
- `docs/dataset/` — Unfallatlas dataset description (`DSB_Unfallatlas.md`/`.pdf`); source of truth for coded categorical labels used in notebook label dictionaries (see `mem:conventions` for the U-Phase label-dict convention).
- `docs/project/` — repo/process docs (`ConventionalCommitsGuide.md`, `PROJEKTPLAN_SETUP.md`).

When moving or adding a doc file: grep the whole repo (`AGENTS.md`, `.serena/memories/`, `docs/prompts/` non-fenced metadata) for the old filename before committing the move — references are not auto-tracked.
