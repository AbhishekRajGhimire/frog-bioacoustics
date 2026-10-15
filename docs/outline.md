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

The validated labels train and evaluate a ResNet-18 detector, which then scans whole folders of recordings and records how likely every five-second window is to hold a frog call.
An analysis step turns those probabilities into calling activity by hour of day and by date, always beside how much was recorded.
An application lets site staff scan their own recordings and see the results without a command line, and a package lets them run it on a Windows computer with nothing installed.

## Current status

Phase 1 made the repository reproducible by standardizing its structure, dependencies, class names, commands, and automated checks.
Phase 2 made the data pipeline trustworthy by validating labels, preventing recording-level leakage, creating deterministic grouped folds, and reporting exactly what enters each evaluation split.

Phase 3 expanded the labels to 1,204 frog clips from 176 recordings and 3,020 background clips from 502 recordings.
Phase 4 trained the first real detector on them.
On recordings it never saw, it finds about 97 of every 100 frog clips and wrongly flags fewer than 1 in 100 background clips, far ahead of the simple baseline.
The result held when whole nights were held out.
Every label comes from one site between January and March, recorded between midnight and noon, so performance elsewhere is still unknown.

Phase 5 made the detector usable on whole folders of recordings.
It scanned all 618 recordings, 51.5 hours of audio, in one to two minutes, and it can be stopped and restarted without losing or repeating work.

Phase 6 is complete, in three parts.
Activity analysis turns a detection record into a report, three figures, and traceable tables with one command.
The application for site staff does the same in a browser: staff choose a folder or memory card, scan it, and see when frogs called, with every number traceable to windows they can listen to.
The package puts the application, its own Python, and the model in one zip file that staff unzip and start with a double-click.
Labeling continues for other sites and seasons, and Phase 7, operational quality, comes next.

See the [roadmap](../roadmap.md) for the complete sequence of project phases.

## The project pipeline

```text
field recordings
    -> five-second audio windows
    -> Mel spectrogram images
    -> careful human labels
    -> validated manifest
    -> recording-group folds and quality reports
    -> baseline and ResNet-18 training
    -> batch detection over whole recordings
    -> activity by hour of day and by date
    -> application for site staff
```

### 1. Raw recordings

Original recordings enter the project under `raw/`.
They are source material and should be treated as immutable.

### 2. Spectrogram generation

`scripts/slice_audio.py` reads recordings and creates one PNG spectrogram for each complete five-second window.
Generated spectrograms are stored under `processed/`.
Each image shows decibels above the recording's own noise floor per frequency band, which keeps faint calls visible and removes steady background such as insect drone.

### 3. Human labeling

The Streamlit labeler shows each spectrogram with its matching audio window.
A confident target call moves to `labeled/litoria_aurea`, a confident background example moves to `labeled/non_target`, and a clip you cannot decide on moves to `labeled/unsure`.
Every press is appended to `labeled/decisions.csv`, and the audit mode replays labeled clips so a decision can be confirmed or changed later.

### 4. Manifest and validation

`scripts/build_manifest.py` validates every label and writes `labeled/manifest.csv`.
The manifest is the source of truth for which examples belong to training, validation, and testing.

### 5. Grouped evaluation

All windows from the same recording stay in one fold and one split.
This prevents a model from being tested on audio that is too closely related to its training data.

### 6. Baseline and model training

`scripts/train_baseline.py` checks that the validated data can pass through a complete training and evaluation path.
`scripts/train_model.py` trains the real detector, a ResNet-18 image model that starts from weights learned on everyday photographs and is fine-tuned on the spectrograms.
It chooses its detection threshold on the validation fold, reports results on the test fold, compares itself with the simple baseline, and repeats the training with each fold held out in turn to show how stable the results are.
Every run is saved as one folder under `models/` with the model, copies of its inputs, every prediction, and a readable `report.md`.
`scripts/verify_model.py` reloads a saved run and proves it reproduces its own predictions.

### 7. Batch detection

`scripts/detect_calls.py` runs a saved model over every recording in a folder and its subfolders.
It renders each recording exactly as the training images were rendered and stores the frog probability of every five-second window.
The results go to one folder under `results/detections/`: a SQLite database that is the complete record, `detections.csv` with every window the model flags, and `recordings.csv` with every recording and its status.
Every row names the source recording, the window, the recorder's clock time, and the exact model and preprocessing used.
A file that cannot be read is recorded as failed instead of stopping the batch, and running the same command again continues where it stopped.

### 8. Activity analysis

`scripts/analyze_activity.py` turns one detection record into calling activity by hour of day and by date.
It reports the share of recorded time in which the model detected a call, always beside how much was recorded, so an hour that was never recorded reads as "not recorded" instead of looking silent.
The results go to one folder under `results/analytics/`: a report, a time-of-day chart, a 24-hour clock chart, a chart by date, and tables.
`windows.csv` lists every window with its date, hour, and probability, so every number in the report can be checked by filtering it.
All times use one fixed UTC offset, so daylight-saving changes cannot shift activity by an hour.

### 9. Application for site staff

`uv run streamlit run classifier_app/app.py --server.address localhost` opens the app in the browser.
Under Scan recordings, a person chooses a folder or memory card, or uploads a few files, names the deployment, and presses Scan.
The scan runs in the background with a progress bar, can be stopped, and continues where it stopped.
Under Results, the person sees the summary, warnings such as recorders whose clock was not set, the three charts, and the threshold table, lists the windows behind any hour or date and listens to them, and downloads everything as one zip file.

Site staff do not need the repository.
`uv run python scripts/build_package.py` builds `dist/FrogCallScanner-<version>-<commit>.zip`, which holds the app, its own Python, every dependency, and the newest saved model, and checks it before keeping it.
Staff unzip it and double-click `Start Frog Call Scanner`; their results go to `%LOCALAPPDATA%\Frog Call Scanner`, so a newer package replaces the old one without losing them.

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
    training.toml              model training settings

  raw/                         immutable source recordings
    external/                  externally supplied recordings
    ponds/                     pond recording collection

  processed/                   reproducible generated data
    external/                  external spectrograms and playback chunks
    ponds/                     pond spectrograms and playback chunks

  labeled/                     human-selected examples and generated manifest
    litoria_aurea/             confident target-call examples
    non_target/                confident non-target examples
    unsure/                    clips the reviewer could not decide on
    decisions.csv              append-only log of every labeling decision

  scripts/                     commands a person runs
    slice_audio.py             generate spectrograms
    sync_labeled_images.py     replace labeled images after regeneration
    verify_spectrograms.py     prove stored images match the contract
    label_frontend.py          Streamlit labeler with label and audit modes
    build_manifest.py          validate labels and build reports
    train_baseline.py          run the strict sanity baseline
    train_model.py             train, evaluate, and save the ResNet-18 detector
    verify_model.py            prove a saved run reproduces its predictions
    detect_calls.py            detect calls in every recording beneath a folder
    analyze_activity.py        summarize calling activity in a detection record
    build_package.py           build the zip that site staff unzip and start

  src/frog_classifier/         reusable Python package
    data/                      config, naming, manifests, folds, validation, and reports
    preprocessing/             Mel rendering, noise floor, PNG encoding, and display colours
    labeling/                  queue order, decision log, moves, and playback audio
    training/                  model, training loop, metrics, saved runs, and verification
    inference/                 recordings, rendering, detection record, and CSV export
    analytics/                 time basis, activity by hour and date, figures, and report
    app/                       app services: deployments, scans, uploads, listening, download, and launcher
    packaging/                 site staff package: contents, build, and self-check

  tests/                       synthetic automated checks
  models/                      saved training runs
  results/                     generated quality reports, detection records, and analytics
  classifier_app/              the Streamlit app for site staff
  dist/                        the built site staff package and its zip

  docs/
    outline.md                 this beginner orientation guide
    architecture.md            implemented components and storage contracts
    workflow.md                detailed operating procedures
    superpowers/               approved design and implementation records
    decisions/                 dated design decisions with their evidence
```

The most important boundary is between `scripts/` and `src/`.
`scripts/` contains runnable entry points.
Phase 2 manifest and validation behavior is factored into `src/frog_classifier/data/`.
New reusable behavior should follow that boundary so it can be imported and tested without running an entire command.

## What Git tracks

Git tracks source code, tests, configuration, documentation, dependency metadata, and `.gitkeep` files that preserve empty directories.

Git normally ignores:

- Raw audio recordings.
- Generated spectrograms and playback chunks.
- Human-labeled image collections.
- Generated manifests and quality reports.
- Trained model artifacts.
- Detection records and analytics outputs.
- Site staff packages under `dist/`.
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

Use Frog only when the target call is confidently present, even if it is faint.
Use Background only when the clip is confidently non-target.
Use Unsure when identification is uncertain, and never convert uncertainty into a negative label.

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
| Spectrogram rendering behavior | `src/frog_classifier/preprocessing/` |
| Labeling queue, decisions, or playback audio | `src/frog_classifier/labeling/` |
| Site staff package or its launcher | `src/frog_classifier/packaging/` or `src/frog_classifier/app/launch.py` |
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
uv run python -B -m py_compile scripts/build_manifest.py scripts/label_frontend.py scripts/slice_audio.py scripts/train_baseline.py
git diff --check
git status --short
```

### 7. Review and commit

Read the diff before staging it.
Stage only intended files.
Replace `docs/outline.md` in the example with every intended path for your change.
After staging, inspect the staged diff and check it before committing.
Use a short commit message that explains the change.

```powershell
git diff
git add docs/outline.md
git diff --cached --check
git diff --cached
git status --short
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
Mark uncertain clips Unsure so they wait for a second hearing instead of becoming background labels.

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
