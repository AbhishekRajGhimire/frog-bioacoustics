# Frog Classifier

Frog Classifier turns long field recordings into fixed five-second Mel-spectrogram examples for careful human review and leakage-safe evaluation of *Litoria aurea* calls.
Each image shows decibels above its own recording's noise floor, so faint calls stay visible and brightness means the same thing in every image.
Phases 1, 2, 4, and 5 are complete: the first ResNet-18 detector is trained, saved, and verified on one site's recordings, and it runs over whole folders of recordings into a resumable, traceable detection record.
Phase 6 is in progress: activity analysis by hour of day and by date and an application for site staff are done, and packaging the application is next.

## Start here

- New to the repository? Read the beginner-friendly [project outline](docs/outline.md) first.
- Read the [roadmap](roadmap.md) for planned phases and acceptance criteria.
- Read the [architecture](docs/architecture.md) for storage contracts and package boundaries.
- Read the [workflow](docs/workflow.md) for day-to-day preprocessing, labeling, manifest, training, and detection commands.

## Pipeline

```text
raw recordings -> spectrogram queue -> human labels -> versioned manifest -> grouped folds and reports -> ResNet-18 training and verification -> batch detection -> activity analysis -> application for site staff
```

`raw/` holds immutable source recordings.
`processed/` holds reproducible spectrograms and optional playback chunks.
`labeled/` holds human-selected PNGs and the generated manifest.
`results/data_quality/` receives generated validation reports.
`models/` receives saved training runs.
`results/detections/` receives detection records.
`results/analytics/` receives activity analyses.
The manifest workflow trains directly from labeled images, so materialized split directories are not part of the tracked layout.

## Setup

Python 3.13 is required.
Install [uv](https://docs.astral.sh/uv/) and create the locked environment:

```powershell
uv sync --frozen
```

On Windows, `Launch_Labeler.bat` provides a graphical-labeler entry point without using uv directly.
It creates `.venv`, installs the locked compatibility export, and opens the Streamlit interface.

## Common commands

Generate spectrograms from the default external recording root:

```powershell
uv run python scripts/slice_audio.py
```

Run a small preprocessing smoke sample:

```powershell
uv run python scripts/slice_audio.py --limit-files 2
```

Process pond recordings separately:

```powershell
uv run python scripts/slice_audio.py --raw-root raw/ponds --out-root processed/ponds/spectrograms
```

Replace the labeled images after regenerating spectrograms:

```powershell
uv run python scripts/sync_labeled_images.py
```

Prove a random sample of stored spectrograms matches the raw audio and the tracked contract:

```powershell
uv run python scripts/verify_spectrograms.py --sample 50
```

Build the validated versioned manifest and data-quality reports:

```powershell
uv run python scripts/build_manifest.py
```

Run the strict logistic-regression baseline after a valid manifest has been generated:

```powershell
uv run python scripts/train_baseline.py
```

Train, evaluate, and save the ResNet-18 detector, then prove the saved run reproduces its predictions:

```powershell
uv run python scripts/train_model.py
uv run python scripts/verify_model.py models/<run folder>
```

Detect calls in every recording beneath a folder with a saved run; running it again continues an interrupted run:

```powershell
uv run python scripts/detect_calls.py models/<run folder> raw/external
```

Summarize calling activity in a detection record by hour of day and by date:

```powershell
uv run python scripts/analyze_activity.py results/detections/<record folder>
```

Scan recordings and explore the results in the browser, without the command line:

```powershell
uv run streamlit run classifier_app/app.py
```

## Labeling policy

Use Frog only when the target call is confidently present, and add the faint tag when it is distant.
Use Background only when the clip is confidently non-target.
Use Unsure when you cannot decide; the clip waits in `labeled/unsure/` and never enters training.
Every decision is appended to `labeled/decisions.csv`, and the labeler's audit mode replays labeled clips so a decision can be confirmed or changed.
