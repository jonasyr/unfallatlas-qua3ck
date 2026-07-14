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

## SVM algorithm selection and binary champion search

**Tool:** Claude Code (Sonnet 5)<br>
**Model release:** June 30, 2026<br>
**Used:** July 2026<br>
**Effort:** Medium<br>
**Disclosure:** [AI TOOL DISCLOSURE.md](../AI%20TOOL%20DISCLOSURE.md)
**Implementation plan:** [2026-07-14-svm-algorithm-selection.md](../superpowers/plans/2026-07-14-svm-algorithm-selection.md)

### Recorded prompt

The plan was produced with `superpowers:writing-plans` from these verbatim
user messages, after the `Einheit 6 – Support Vector Machines` course
material was written and used as the review baseline:

> Perfect thank you!!! Now please based on the source material / the newly
> created docs/course-material/Einheit 6 – Support Vector Machines.md
> review carefully our A Phase if we really did everything we could/should
> to get the best performance possible without Bias/over/underfitting etc.
> Especially review the Use of SVM and everything else that is described
> here like kernel modules etc. and implement what can be / should be SVM
> is a MUST (read the source files). Use /writing-plans to plan this out
> carefulyl so nothing breaks and implement it with subagent driven
> development. ALso one small thing you have a nice visualization here
> [screenshot of the 3-class Pareto-front plot] but wouldnt it be cooler
> to have the Final Models with binary classification in it too or an
> extra one for them? because we have nothing visual for the actual
> working binary stuff do we?

> ok but jisz a question why not do a new champion finder for the binary
> classification and just assume the one that performs best on the 3
> class classification is the best for binary? I have no experience
> genuine question and the seconf thought is why is A3 Notebook basically
> finished at 10 and then a new §9 emerges. Generally i get your point but
> it just looks and feels a bit taped together as quick fix and not a
> professional scientific notebook. Do you know what i mean? maybe improve
> the plan a bit based on that?

> LETS GO IMPLEMENT IT!!! In the end update
> @docs/prompts/03_prompts_phase_a3.md and
> @"docs/AI TOOL DISCLOSURE.md"

The second message above changed the plan's scope before implementation
began: instead of assuming the 3-class champion also wins the binary
KSI reframing, the plan was revised to run a genuine Stage 0 (baseline)
/ Stage 1 (ten-candidate, including all three SVM variants) champion
search for the binary target, and to fix a duplicate section-numbering
defect (`## §9` recurring after `## 10`) that made the notebook look
patched together rather than authored as one coherent document.

The plan was implemented with `superpowers:subagent-driven-development`
across seven tasks: a gate-aware binary decision-threshold search helper,
three SVM pipeline builders (linear soft-margin, hinge-loss SGD, RBF
kernel), a binary-specific Pareto-front visualization, the
section-numbering fix, the genuine binary champion search itself, Optuna
tuning/refit/threshold-selection/Test-2024 evaluation of the winning
family, and a final end-to-end notebook execution with the results
summary filled in from the real run. Two real, previously-latent bugs
were found and fixed via live execution rather than static review: an
XGBoost binary-classification `IndexError` (the 3-class builder's
`objective="multi:softprob", num_class=3` doesn't apply to a 2-class
target) and an Optuna progress-callback crash on pruned/failed trials
(`trial.value` is `None` in that case). The binary champion search
selected `random_forest` over nine other candidates (including all three
SVM variants), reaching Test-2024 macro-F1 = 0.6026 and recall(KSI) =
0.5255, clearing both acceptance gates.

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
