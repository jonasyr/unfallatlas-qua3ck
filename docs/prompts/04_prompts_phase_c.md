# AI Prompt Records — Phase C

Each record preserves the available prompt or planning context. Detailed
implementation instructions live only in the linked plan files.

## Course-material comparison, C-phase design, and full notebook build

**Tool:** Claude Opus 4.7 (1M context)<br>
**Model release:** April 16, 2026<br>
**Used:** July 2026<br>
**Effort:** Medium<br>
**Disclosure:** [AI TOOL DISCLOSURE.md](../AI%20TOOL%20DISCLOSURE.md)<br>
**Design spec:** [2026-07-21-c-phase-notebook-design.md](../superpowers/specs/2026-07-21-c-phase-notebook-design.md)<br>
**Implementation plan:** [2026-07-21-c-phase-notebook.md](../superpowers/plans/2026-07-21-c-phase-notebook.md)

### Recorded prompt

The original request, before being rewritten into the structured form the
session actually followed:

> Do me a favour and do the following: Check docs/course-material inside my
> and my friends portfolio and check whoevers notes are more detailed in
> depth and generally better. If my friends are better replace mine with his
> if mine are better do nothing. Then the next task would be the following.
> Read exactly the Task description and scope of the different phases and
> most importantly my results from my repo HTML Notebooks etc reports etc
> etc. Then check the different notebooks inside
> @Degrees-of-No-Return-App/ and especially the C Phase. After all of that
> and only then plan out the creation of the C Notebook via /writing-plans
> for the beforehand stuff you could use /brainstorming. The goal in this
> session is to have a finished PERFECT C Phase Notebook set and done. Then
> afterwards update @"unfallatlas-qua3ck/docs/AI TOOL DISCLOSURE.md" and
> @unfallatlas-qua3ck/docs/prompts/ based on the existing structure with the
> plan/prompt from this session. As first step improve this prompt by
> rewriting it via prompt best practices. Use serena and codebasemem for
> inside @unfallatlas-qua3ck/

The prompt was rewritten into an explicit, phase-ordered structure (course-
material comparison → assignment/results research → reference-solution
review → `/brainstorming` design → `/writing-plans` implementation plan →
notebook build → disclosure update) and presented back for confirmation
before any file changes were made.

> Lets GO!

> implement it fully your on auto mode just give me a summary at the very
> end but fully do this on your own

This second message authorized unattended execution for the remainder of
the session — the plan below was executed inline (not via subagent
dispatch, since the analysis is a single tightly-sequential notebook build)
without further check-ins, including the disclosure-update step.

> but keep the cramers v from the a phase at the end with binary
> classification in binary classification in general

Mid-build correction: the C-phase §6 literature-alignment text was made to
explicitly distinguish the binary-target Cramér's V (`UART`=0.1801, computed
in A³ §20 against the binary KSI label) from the earlier 3-class ceiling's
Cramér's V (≤0.13, against the original `UKATGEORIE`), rather than leaving
the binary/3-class distinction implicit.

### Planning context

**Course-material comparison:** `docs/course-material/` in this repo and in
`EnergyCast-App/docs/course-material/` were compared unit-by-unit (word
counts, heading structure, and content depth). The friend's notes
consistently included a table of contents, a "Key terms" glossary, and a
"Source and validation check" section in all 7 units — none of my notes had
any of these — and were 44% longer on the most complex unit (SVM). My notes
were replaced with the friend's set; a supplementary note with no equivalent
in the friend's repo (`Master Data Analysis with ChatGPT.md`) was kept
unchanged.

**Design and planning:** `superpowers:brainstorming` settled the C-phase
scope (full `PROJEKTPLAN_SETUP.md` scope, not the narrower, partially-stale
A³ §10 handoff note), the SHAP depth (global + 4 local case examples), how
the literature-alignment section should build on A³ §20 rather than
duplicate it, whether to include a qualitative weighted evaluation matrix
(yes), and what the classification-appropriate replacement for the
reference solution's regression-specific "residual analysis" section should
be (error-slice diagnostics by accident type / road context / weather).
`superpowers:writing-plans` then produced the task-by-task implementation
plan linked above.

**Implementation:** Executed inline across 8 tasks: two library-helper
tasks (`src/unfallatlas/models/c_phase.py` for error-slice diagnostics,
qualitative-matrix scoring, and the K-phase inference-contract builder;
`plot_roc_pr_curves`/`plot_confusion_matrix_heatmap` added to
`src/unfallatlas/viz/metrics_viz.py`, both TDD'd against new/extended test
files), then the notebook's 11 sections (§0 setup and artifact loading
through the closing summary), each synced via `jupytext --sync`, executed
end-to-end via `jupyter nbconvert --execute`, and committed once its output
was read back and any narrative placeholders were replaced with the actual
computed numbers.

One implementation-time correction: the initial SHAP `TreeExplainer` call
did not terminate within 10 minutes on the champion's very deep trees (180
trees, depth 23, ~7.4M total leaves) — confirmed via a standalone benchmark
script rather than assumed. Switching to `approximate=True` (Saabas
algorithm) reduced a 500-row run to 0.11 seconds; this was adopted and the
reasoning (with the empirical timing) was documented directly in the
notebook's §5 markdown rather than silently applied.

The final notebook execution passed a placeholder scan (no `[PLACEHOLDER`,
`[to be filled`, or `TBD` markers remaining) and the full project test
suite (317 tests).
