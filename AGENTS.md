# Project Rules for AI Agents

## Project purpose

Frog Bioacoustics is building a trustworthy system for finding and analyzing *Litoria aurea* calls in field recordings.
The pipeline converts recordings into five-second Mel spectrograms, supports careful human labeling, validates evaluation data, and will later support model training, batch detection, and activity analysis.

Phases 1 and 2 are complete.
Phase 3, purposeful label expansion, is the next project priority.
The current model baseline verifies the pipeline and is not a production detector.

## Start with these documents

- Read `docs/outline.md` for a beginner-friendly project overview.
- Read `docs/architecture.md` for implemented boundaries and storage contracts.
- Read `docs/workflow.md` for operational commands.
- Read `roadmap.md` for project priorities and acceptance criteria.
- Treat `config/preprocessing.toml` as the machine-readable preprocessing and class contract.

Do not duplicate detailed documentation when an authoritative document already exists.
Update the authoritative document when its behavior changes.

## Project structure

- `scripts/` contains runnable command entry points.
- `src/frog_classifier/` contains reusable Python behavior.
- `config/` contains shared machine-readable configuration.
- `tests/` contains synthetic automated verification.
- `raw/` contains immutable source recordings.
- `processed/` contains regenerable spectrograms and playback chunks.
- `labeled/` contains human-selected examples, the unsure holding folder, the decision log, and the generated manifest.
- `models/` and `results/` contain generated artifacts.
- `docs/` contains project guidance, architecture, workflow, and planning history.

Keep command wrappers focused.
Place new reusable behavior in `src/frog_classifier/` behind clear, tested interfaces.

## Clean-code standards

- Write simple, readable, maintainable code.
- Prefer small units with one clear responsibility.
- Use clear names that describe purpose rather than implementation details.
- Keep interfaces explicit and avoid hidden side effects.
- Avoid duplicated logic and unnecessary abstraction.
- Validate inputs at clear boundaries and provide actionable errors.
- Preserve existing behavior unless the requested change intentionally modifies it.
- Do not refactor unrelated code during a focused task.
- Follow the existing project layout and established patterns.

## Comments

Write professional comments when they explain intent, constraints, invariants, safety risks, or a non-obvious decision.
Keep comments concise and accurate.
Do not add comments that merely restate the code.
Update or remove comments when behavior changes.
Prefer clear code over comments that compensate for confusing structure.

## Data and evaluation safety

- Treat raw recordings as immutable source material.
- Never delete, overwrite, rename, or normalize private data unless the user explicitly requests it.
- Preserve human labels because they are expensive and non-regenerable.
- Use Frog only when the target call is confidently present.
- Use Background only when the clip is confidently non-target.
- Mark uncertain examples Unsure instead of converting uncertainty into a negative label; unsure clips never enter training.
- Keep every recording group in one evaluation fold and split.
- Use the validated manifest as the source of truth for training and evaluation membership.
- Do not commit recordings, generated spectrograms, labels, manifests, reports, models, or analytics without an explicit storage decision.
- Inspect exact paths before any destructive filesystem action.

## Testing and verification

- Reproduce a bug through the closest practical user-facing path before fixing it.
- Add or update focused tests when behavior changes.
- Use synthetic temporary data so automated tests do not require private recordings or labels.
- Run the smallest relevant test while iterating.
- Run broader verification in proportion to the risk of the change.
- Inspect visible user interfaces when changing user-facing behavior.
- Fix failures caused by the current change and report unrelated failures.
- Check `git diff` and `git status --short` before handing work back.
- Do not claim completion without fresh verification evidence.

## Documentation work

Handle routine documentation changes directly and keep the process simple.
Do not create a design specification, implementation plan, or extra planning artifact for a routine documentation edit unless the user explicitly asks for one.
Do not invoke optional planning workflows for routine documentation edits unless they are explicitly requested.
Higher-priority system and developer instructions still apply when they require a specific workflow or skill.

Keep documentation accurate, beginner-friendly, and linked to authoritative sources.
Put each full sentence on its own physical Markdown line when writing or substantially editing long Markdown files.
Do not use the em dash character.

## Git behavior

Keep changes local by default.
Do not commit, push, merge, create a pull request, rename a remote repository, or delete a branch unless the user explicitly asks.
Do not change branches when a local edit can be completed safely in the current checkout.
Preserve unrelated user changes in a dirty worktree.
Never use destructive Git commands such as `git reset --hard` without explicit authorization.

## Scope and communication

Make reasonable assumptions that keep work moving without expanding the requested scope.
Ask only when a missing decision would materially change the result or create risk.
Lead with the outcome and explain technical details only when they help the user verify or continue the work.
Report remaining limitations honestly, especially the small number of confirmed positive frog examples.
