# Beginner Project Outline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a beginner-friendly `docs/outline.md` that explains the project, repository structure, daily workflows, safety rules, and first-contribution process.

**Architecture:** The new guide will be the orientation layer between the concise root README and the detailed architecture, workflow, and roadmap documents.
The repository navigation test will require the guide, validate its relative links, and require a discoverable README link so documentation drift becomes a test failure.

**Tech Stack:** GitHub-flavored Markdown, PowerShell command examples, Python 3.13, `uv`, and Python `unittest` repository checks.

## Global Constraints

- Write for a beginner who may not know bioacoustics, machine learning, Python project layouts, or this repository.
- Keep every full sentence on its own physical Markdown line.
- Do not use the em dash character.
- Use repository-relative Markdown links that resolve from `docs/outline.md`.
- Keep detailed storage contracts in `docs/architecture.md`, operational procedures in `docs/workflow.md`, and roadmap acceptance criteria in `roadmap.md`.
- Do not rename the Python distribution, local folder, package imports, scripts, or existing documents.
- Do not change dependencies, application behavior, preprocessing values, data, labels, models, or generated artifacts.
- Treat raw recordings as immutable source material.
- Treat uncertain examples as skipped work rather than negative labels.
- Preserve the manifest as the source of truth for grouped training and evaluation membership.

---

### Task 1: Add the tested beginner orientation guide

**Files:**

- Create: `docs/outline.md`
- Modify: `README.md`
- Modify: `tests/test_repository_reproducibility.py`

**Interfaces:**

- Consumes: the verified project facts and commands in `README.md`, `docs/architecture.md`, `docs/workflow.md`, `roadmap.md`, and `config/preprocessing.toml`.
- Produces: a beginner orientation guide at `docs/outline.md` and a root README link labeled `project outline`.
- Verifies: `OPERATIONAL_DOCUMENTS` and `OPERATIONAL_TEXT_FILES` include `docs/outline.md`, all relative Markdown targets exist, and the README exposes the guide.

- [ ] **Step 1: Write the failing navigation test**

Modify `tests/test_repository_reproducibility.py` so the relevant constants and the new test read as follows:

```python
OPERATIONAL_DOCUMENTS = (
    "README.md",
    "roadmap.md",
    "docs/outline.md",
    "docs/architecture.md",
    "docs/workflow.md",
)

OPERATIONAL_TEXT_FILES = (
    "README.md",
    "docs/outline.md",
    "docs/architecture.md",
    "docs/workflow.md",
    "scripts/build_manifest.py",
    "scripts/label_spectrograms.py",
    "scripts/slice_audio.py",
    "scripts/train_baseline.py",
)
```

Add this method to `RepositoryLayoutTests` after `test_project_navigation_files_are_complete_and_linked`:

```python
    def test_beginner_outline_is_discoverable(self) -> None:
        outline_path = REPO_ROOT / "docs/outline.md"
        self.assertTrue(outline_path.is_file())
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("[project outline](docs/outline.md)", readme)
```

- [ ] **Step 2: Run the focused test to verify RED**

Run:

```powershell
uv run python -m unittest tests.test_repository_reproducibility -v
```

Expected: the suite fails because `docs/outline.md` does not exist and the README does not yet contain `[project outline](docs/outline.md)`.

- [ ] **Step 3: Add the README entry point**

Replace the opening of the `README.md` `Start here` list with:

```markdown
## Start here

- New to the repository? Read the beginner-friendly [project outline](docs/outline.md) first.
- Read the [roadmap](roadmap.md) for planned phases and acceptance criteria.
- Read the [architecture](docs/architecture.md) for storage contracts and package boundaries.
- Read the [workflow](docs/workflow.md) for day-to-day preprocessing, labeling, manifest, and baseline commands.
```

- [ ] **Step 4: Create the complete beginner guide**

Create `docs/outline.md` with this complete content:

````markdown
# Frog Bioacoustics Project Outline

This guide explains what the project does, how the repository is organized, and how to work on it safely.
It is written for someone opening the project for the first time.

The GitHub repository is named Frog Bioacoustics.
The installed Python distribution is still named `frog-classifier`, and its import package is `frog_classifier`.

## What the project is

The project is building a system that can find *Litoria aurea* frog calls in long field recordings.
Field recordings are difficult to review directly because each file can contain minutes or hours of sound.
The project divides recordings into five-second windows and turns each window into a Mel spectrogram, which is an image showing how sound energy changes across frequency and time.

A person reviews those images and matching audio clips.
Confident target calls become `litoria_aurea` examples, confident non-target sounds become `non_target` examples, and uncertain clips are skipped.

The validated labels can later train and evaluate a model.
Future phases will add stronger model training, batch detection, activity analysis, and a user-facing application.

## Current status

Phase 1 made the repository reproducible by standardizing its structure, dependencies, class names, commands, and automated checks.
Phase 2 made the data pipeline trustworthy by validating labels, preventing recording-level leakage, creating deterministic grouped folds, and reporting exactly what enters each evaluation split.

Phase 3 is next.
It will expand positive examples and difficult negative examples across independent recordings and field conditions.

The current local acceptance set contains 13 confirmed positive examples and 48 confirmed negative examples.
That is enough to verify the pipeline, but it is not enough to claim stable model performance.
The logistic-regression baseline is therefore a pipeline sanity check rather than a production detector.

See the [roadmap](../roadmap.md) for the complete sequence of project phases.

## The project pipeline

```text
field recordings
    -> five-second audio windows
    -> Mel spectrogram images
    -> careful human labels
    -> validated manifest
    -> recording-group folds and quality reports
    -> baseline and future model training
    -> future detections and activity analysis
```

### 1. Raw recordings

Original recordings enter the project under `raw/`.
They are source material and should be treated as immutable.

### 2. Spectrogram generation

`scripts/slice_audio.py` reads recordings and creates one PNG spectrogram for each complete five-second window.
Generated spectrograms are stored under `processed/`.

### 3. Human labeling

The labelers show each spectrogram with its matching audio window.
A confident target call moves to `labeled/litoria_aurea`, while a confident background example moves to `labeled/non_target`.

### 4. Manifest and validation

`scripts/build_manifest.py` validates every label and writes `labeled/manifest.csv`.
The manifest is the source of truth for which examples belong to training, validation, and testing.

### 5. Grouped evaluation

All windows from the same recording stay in one fold and one split.
This prevents a model from being tested on audio that is too closely related to its training data.

### 6. Baseline and future models

`scripts/train_baseline.py` checks that the validated data can pass through a complete training and evaluation path.
Later phases will add reproducible saved models and batch recording inference.

## Repository structure

```text
frog-bioacoustics/
  README.md                    short project entry point
  roadmap.md                   phases, priorities, and acceptance criteria
  pyproject.toml               Python project and direct dependencies
  uv.lock                      exact locked dependency environment
  Launch_Labeler.bat           Windows graphical-labeler launcher

  config/
    preprocessing.toml         audio, spectrogram, rendering, and class contract

  raw/                         immutable source recordings
    external/                  externally supplied recordings
    ponds/                     pond recording collection

  processed/                   reproducible generated data
    external/                  external spectrograms and playback chunks
    ponds/                     pond spectrograms and playback chunks

  labeled/                     human-selected examples and generated manifest
    litoria_aurea/             confident target-call examples
    non_target/                confident non-target examples

  scripts/                     commands a person runs
    slice_audio.py             generate spectrograms
    label_frontend.py          Streamlit labeling interface
    label_spectrograms.py      Matplotlib labeling interface
    build_manifest.py          validate labels and build reports
    train_baseline.py          run the strict sanity baseline

  src/frog_classifier/         reusable Python package
    data/                      config, naming, manifests, folds, validation, and reports

  tests/                       synthetic automated checks
  models/                      future generated model artifacts
  results/                     generated quality and analytics outputs
  classifier_app/              future user-facing application

  docs/
    outline.md                 this beginner orientation guide
    architecture.md            implemented components and storage contracts
    workflow.md                detailed operating procedures
    superpowers/               approved design and implementation records
```

The most important boundary is between `scripts/` and `src/`.
Files in `scripts/` are thin command entry points.
Reusable behavior belongs in `src/frog_classifier`, where it can be imported and tested without running an entire command.

## What Git tracks

Git tracks source code, tests, configuration, documentation, dependency metadata, and `.gitkeep` files that preserve empty directories.

Git normally ignores:

- Raw audio recordings.
- Generated spectrograms and playback chunks.
- Human-labeled image collections.
- Generated manifests and quality reports.
- Trained model artifacts.
- Analytics outputs.
- Local Python environments such as `.venv`.

Ignored does not mean unimportant.
Human labels are expensive, non-regenerable work and need a separate backup strategy.
Raw recordings must also be protected outside normal Git history.

Before committing, always run `git status --short` and check every listed path.

## Getting started

### Requirements

- Python 3.13.
- Git.
- `uv` for the locked Python environment.
- A Windows environment if using `Launch_Labeler.bat`.

Clone the repository and enter it:

```powershell
git clone https://github.com/AbhishekRajGhimire/frog-bioacoustics.git
Set-Location frog-bioacoustics
```

Create or update the exact locked environment:

```powershell
uv sync --frozen
```

Run the small repository check:

```powershell
uv run python -m unittest tests.test_repository_reproducibility -v
```

Run the complete test suite when you need to verify the whole project:

```powershell
uv run python -m unittest discover -s tests -v
```

## Working with the data pipeline

Run commands from the repository root.
Read the detailed [workflow](workflow.md) before operating on the full data collection.

### Generate spectrograms

Process the default external recording tree:

```powershell
uv run python scripts/slice_audio.py
```

Use a small smoke sample before a full run:

```powershell
uv run python scripts/slice_audio.py --limit-files 2
```

Generated files go under `processed/` and should not be committed.

### Label examples

On Windows, double-click `Launch_Labeler.bat` to open the graphical labeler.

The alternative Matplotlib labeler is:

```powershell
uv run python scripts/label_spectrograms.py --limit 12 --shuffle
```

Use Frog only when the target call is confidently present, even if it is faint.
Use Background only when the clip is confidently non-target.
Use Skip when identification is uncertain, and never convert uncertainty into a negative label.

### Build the validated manifest

After labels are available, run:

```powershell
uv run python scripts/build_manifest.py
```

This writes the ignored manifest at `labeled/manifest.csv` and quality reports under `results/data_quality/`.
The command fails clearly when labels, paths, classes, folds, or output destinations are unsafe.

### Run the strict baseline

After building a valid manifest, run:

```powershell
uv run python scripts/train_baseline.py
```

The baseline prints metrics for train, validation, and test splits.
It does not save a production model.

## Where to make a change

Use this guide when deciding which file to edit:

| Change | Location |
| --- | --- |
| Command options or command orchestration | `scripts/` |
| Reusable data behavior | `src/frog_classifier/data/` |
| Audio and spectrogram defaults | `config/preprocessing.toml` |
| Automated verification | `tests/` |
| Beginner entry points | `README.md` or `docs/outline.md` |
| Detailed operating instructions | `docs/workflow.md` |
| Implemented system boundaries | `docs/architecture.md` |
| Future sequencing and acceptance criteria | `roadmap.md` |

Keep command wrappers small.
If behavior needs to be reused or tested independently, place it in the package and call it from the script.

## Safe contribution workflow

### 1. Update your local repository

```powershell
git switch main
git pull --ff-only origin main
```

### 2. Create a focused branch

```powershell
git switch -c feature/short-description
```

Choose a name that describes one bounded change.

### 3. Read the relevant contract

Read [architecture.md](architecture.md) before changing component boundaries or storage behavior.
Read [workflow.md](workflow.md) before changing operational commands.
Read [preprocessing.toml](../config/preprocessing.toml) before changing audio, image, or class settings.

### 4. Make one scoped change

Do not modify unrelated behavior while completing a focused task.
Preserve existing user work and private data.

### 5. Add or update tests

Use synthetic temporary data so tests do not require the private recording collection.
Bug fixes should first reproduce the user-visible failure as closely as practical.

### 6. Run verification

Start with the most focused relevant test, then run:

```powershell
uv lock --check
uv sync --frozen
uv pip check
uv run python -m unittest discover -s tests -v
uv run python -B -m py_compile scripts/build_manifest.py scripts/label_frontend.py scripts/label_spectrograms.py scripts/slice_audio.py scripts/train_baseline.py
git diff --check
git status --short
```

### 7. Review and commit

Read the diff before staging it.
Stage only intended files and use a short commit message that explains the change.

```powershell
git diff
git add docs/outline.md
git commit -m "docs: improve beginner project outline"
```

## First-contribution checklist

- [ ] I understand which project phase my change supports.
- [ ] I read the relevant outline, workflow, architecture, or roadmap section.
- [ ] I created or updated the locked environment with `uv sync --frozen`.
- [ ] I created a focused branch from an up-to-date `main`.
- [ ] I changed only the files required for the task.
- [ ] I added or updated synthetic tests when behavior changed.
- [ ] I ran the focused test for my change.
- [ ] I ran the complete repository test suite.
- [ ] I ran `git diff --check`.
- [ ] I inspected `git status --short` for private or generated files.
- [ ] I read the final diff before committing.

## Important rules and common mistakes

### Protect source data

Do not delete, rename, rewrite, or normalize raw recordings as a side effect of development.
Test data-changing commands on a small explicit sample first.

### Do not turn uncertainty into a negative label

An unclear or distant call is not reliable background evidence.
Skip uncertain clips until Phase 3 adds an explicit review-later state and signal-quality metadata.

### Keep recording groups together

Do not manually create random image-level train and test directories.
Use the validated manifest and grouped folds so one recording cannot leak across evaluation boundaries.

### Do not edit generated dependency exports manually

`pyproject.toml` declares direct dependencies.
`uv.lock` records the exact environment.
`requirements_labeler.txt` is generated for the Windows launcher and should be refreshed from the lock rather than edited by hand.

### Do not commit private or generated artifacts

The `.gitignore` rules protect common paths, but they do not replace reviewing `git status` and the staged diff.
Never force-add recordings, generated images, labels, models, or results without an explicit storage decision.

### Do not trust a metric without its data report

Always confirm class counts, recording-group counts, folds, and splits before interpreting model metrics.
The current small positive set makes validation results unstable even though the pipeline is structurally correct.

## Further reading

- [README](../README.md) for the shortest project entry point.
- [Architecture](architecture.md) for implemented components and storage contracts.
- [Workflow](workflow.md) for detailed operating commands.
- [Roadmap](../roadmap.md) for project phases and acceptance criteria.
- [Preprocessing configuration](../config/preprocessing.toml) for machine-readable audio, spectrogram, rendering, and class settings.
- [Tests](../tests/) for executable examples of expected behavior.
````

- [ ] **Step 5: Run the focused test to verify GREEN**

Run:

```powershell
uv run python -m unittest tests.test_repository_reproducibility -v
```

Expected: seven repository reproducibility tests pass, including the new outline-discoverability test and relative-link checks.

- [ ] **Step 6: Review the guide against the approved design**

Confirm each of these statements directly in `docs/outline.md`:

- A beginner can explain the project and its current Phase 3 handoff.
- The pipeline and every top-level working directory have plain-language explanations.
- Tracked, generated, private, regenerable, and non-regenerable content are distinguished.
- Setup, preprocessing, labeling, manifest, baseline, and test commands match the existing scripts.
- The guide explains where command, package, configuration, test, and documentation changes belong.
- The contribution workflow and checklist include branch, test, diff, status, and commit steps.
- Raw-data protection, uncertain-label handling, grouped evaluation, dependency-export handling, and metric limitations are explicit.
- All deeper links point to the existing authoritative documents.

- [ ] **Step 7: Run complete verification**

Run:

```powershell
uv lock --check
uv sync --frozen
uv pip check
uv run python -m unittest discover -s tests -v
uv run python -B -m py_compile scripts/build_manifest.py scripts/label_frontend.py scripts/label_spectrograms.py scripts/slice_audio.py scripts/train_baseline.py
rg -n "TBD|TODO|FIXME|PLACEHOLDER|\?\?\?" docs/outline.md
rg --pcre2 --line-number "\x{2014}" docs/outline.md README.md
git diff --check
git status --short
```

Expected: the dependency and compilation checks pass, 84 tests pass, both `rg` commands return no matches, `git diff --check` returns no output, and Git lists only `README.md`, `docs/outline.md`, and `tests/test_repository_reproducibility.py` as implementation changes.

- [ ] **Step 8: Commit the guide**

Run:

```powershell
git add README.md docs/outline.md tests/test_repository_reproducibility.py
git diff --cached --check
git commit -m "docs: add beginner project outline"
```

Expected: one documentation commit containing only the guide, its README entry point, and its regression coverage.

---

## Completion evidence

Report all of the following before claiming success:

- The exact `docs/outline.md` path and commit.
- The focused repository-test count with zero failures.
- The complete repository-test count with zero failures.
- Successful dependency, compilation, placeholder, prohibited-character, Markdown-link, and Git diff checks.
- Confirmation that no raw, processed, labeled, model, result, or dependency artifact changed.
