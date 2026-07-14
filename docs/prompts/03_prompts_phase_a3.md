# AI Prompt Records — Phase A³

Each record preserves the available prompt or planning context. Detailed
implementation instructions live only in the linked plan files.

## A³ modelling plan and build-out

**Tool:** Claude Code (Sonnet 5)<br>
**Model release:** June 30, 2026<br>
**Used:** July 2026<br>
**Effort:** Medium<br>
**Disclosure:** [AI TOOL DISCLOSURE.md](../AI%20TOOL%20DISCLOSURE.md)
**Implementation plan:** [2026-07-01-a3-phase-modelling.md](../superpowers/plans/2026-07-01-a3-phase-modelling.md)

### Recorded prompt

The plan was produced with `superpowers:writing-plans` from these verbatim
user messages:

> Now please do me a favour now the A Phase writing creation and
> implementation is the next step and i want you to first read all the
> @docs/ and especiall the relevant ones additionally the already existing
> Q and U Phase as it builds upon these 2 then craft a veeery detailed plan
> using /writing-plans where you make AN EXACT SCOPE what the exact things
> are that HAVE to be done in A Phase and what explicitly DONT HAS to go in
> the A PHase as i had issues previously about exactly this in Q and U
> phase with you so make this extra good. Then utilize all the information
> to make a extremly strong A Phase tailored perfectly to my project and
> its context. Then display the plan to me so i can approve it

> IMPORTANT write the finished approved plan as first task inside
> @docs/prompts/ folder as a new prompt file this one for A phase and
> update @"docs/AI TOOL DISCLOSURE.md"

The plan was implemented with `superpowers:subagent-driven-development`.

## Recall-aware champion pivot

**Tool:** Claude Code (Sonnet 5)<br>
**Model release:** June 30, 2026<br>
**Used:** July 2026<br>
**Effort:** Medium<br>
**Disclosure:** [AI TOOL DISCLOSURE.md](../AI%20TOOL%20DISCLOSURE.md)
**Implementation plan:** [2026-07-06-a3-champion-pivot.md](../superpowers/plans/2026-07-06-a3-champion-pivot.md)

### Planning context

After the original validation rule selected `random_forest_balanced` despite
recall for class 1 falling below the acceptance gate, the planning session
changed the candidate-selection rule to carry CatBoost and LightGBM forward.
The plan was produced with `superpowers:brainstorming` and
`superpowers:writing-plans`, then implemented with
`superpowers:subagent-driven-development`.

## OSM feature integration

**Tool:** Claude Code (Sonnet 5)<br>
**Model release:** June 30, 2026<br>
**Used:** July 2026<br>
**Effort:** Medium<br>
**Disclosure:** [AI TOOL DISCLOSURE.md](../AI%20TOOL%20DISCLOSURE.md)
**Implementation plan:** [2026-07-09-a3-osm-feature-integration.md](../superpowers/plans/2026-07-09-a3-osm-feature-integration.md)

### Planning context

This follow-up consumed the U-phase OSM/H3 road-context features in the A³
preprocessor and addressed the three review findings deferred by the
champion-pivot work. It was planned with `superpowers:writing-plans` and
implemented with `superpowers:subagent-driven-development`.

## Binary KSI reframe

**Tool:** Claude Code (Sonnet 5)<br>
**Model release:** June 30, 2026<br>
**Used:** July 2026<br>
**Effort:** Medium<br>
**Disclosure:** [AI TOOL DISCLOSURE.md](../AI%20TOOL%20DISCLOSURE.md)
**Implementation plan:** [2026-07-14-binary-ksi-reframe.md](../superpowers/plans/2026-07-14-binary-ksi-reframe.md)

### Planning context

The recorded plan replaces the infeasible three-class acceptance target with
a binary KSI target and threads the empirical rationale through the Q, U,
and A³ notebooks. No separate verbatim user prompt transcript was retained.

## AI-tool disclosure and prompt-record audit

**Tool:** Codex (GPT-5.6 Terra)<br>
**Model release:** July 9, 2026<br>
**Used:** July 2026<br>
**Effort:** Medium<br>
**Disclosure:** [AI TOOL DISCLOSURE.md](../AI%20TOOL%20DISCLOSURE.md)

### Consolidated prompt

```markdown
Audit and correct the AI-use provenance for this repository as a scientific
portfolio artefact. Work only on `docs/AI TOOL DISCLOSURE.md` and the three
files in `docs/prompts/`; do not change notebooks, source code, data, or
implementation-plan files.

Read the complete disclosure, every prompt record, every file under
`docs/superpowers/plans/`, and the relevant Git history before editing.
Treat Git commits as the project-use evidence because work was committed
immediately after each task. Do not infer a model-use date from a model
release date, and do not invent or add prompt records that were not already
recorded.

Produce a complete and internally consistent documentation system:

1. Keep the detailed implementation plans in `docs/superpowers/plans/`.
   In prompt records and the disclosure, link to a plan by its
   repository-relative path instead of copying its task-by-task content.
   Ensure every repository plan is linked at least once.
2. Give all existing prompt entries the same metadata shape: tool, model
   release date, documented project-use month, effort, disclosure link, and
   plan link when a plan exists. Preserve verbatim historical prompt text.
3. Use Git history to estimate only the use month for each already-recorded
   prompt: May 2026 for the established Q/U work, July 2026 for the existing
   Sonnet 5 A³/OSM work, and July 2026 for this Codex documentation audit.
   State use dates as `Used: <month> 2026` and keep exact commit dates only
   where they are a plan-record date or data-retrieval date.
4. In the disclosure table, use one consistent `Record / plan` link-label
   style: `Prompt record`, `Implementation plan`, a specifically named plan,
   or `This disclosure`. Do not use raw paths or inconsistent labels such as
   `view`.
5. Maintain a complete plan index with all plan files, their recorded dates,
   scope, model/effort, and working links. Do not create a new plan merely
   for this audit.
6. Make the bibliography APA 7 compliant. Cite official first-party model
   release pages using their exact publication dates, distinct from the
   project-use months. Cite the DWD dynamic dataset with `n.d.` and its known
   retrieval date of May 18, 2026. Do not use directory-listing timestamps as
   publication dates.
7. Attribute this audit to `Codex (GPT-5.6 Terra), effort: medium; used July
   2026`, record it as Phase A³ work, and link this prompt record from the
   disclosure table.

Before finishing, verify that all plan links resolve, every plan file is
referenced, all disclosure and prompt metadata agree, no copied plan body
remains in a prompt record, and `git diff --check -- docs` passes. Report
only the files changed and the validation results.
```
