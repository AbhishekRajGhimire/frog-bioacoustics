# Frog Classifier Process Log

This document records what the project currently does, how to reproduce it, and the operational decisions that must remain stable.
The root [roadmap](roadmap.md) tracks future work and completion criteria.

## Current checkpoint

The current local dataset contains 618 five-minute recordings.
All recordings have been converted into 37,080 five-second spectrogram examples.
There are 37,019 spectrograms still awaiting review, 13 labeled `litoria_aurea` examples, and 48 labeled `non_target` examples.
No model or inference result has been saved yet.

The preprocessing stage is complete for the external recording set.
The active work is data-quality and evaluation hardening followed by purposeful expansion of the labeled set.

## Canonical repository layout

```text
frog-classifier/
  raw/
    external/                     # original external recordings
    ponds/                        # original pond recordings
  processed/
    external/
      spectrograms/               # generated five-second Mel-spectrogram PNGs
      chunks/                     # cached five-second WAV files for playback
    ponds/
      spectrograms/
      chunks/
  labeled/
    litoria_aurea/                # positive target examples
    non_target/                   # negative examples
    manifest.csv                  # grouped train, validation, and test assignments
  dataset/
    train/{litoria_aurea,non_target}/
    val/{litoria_aurea,non_target}/
    test/{litoria_aurea,non_target}/
  models/                         # saved model artifacts
  results/analytics/              # inference and analysis outputs
  classifier_app/                 # reserved application package
  scripts/                        # current command-line and labeling tools
  docs/                           # design and project documentation
```

Raw audio and generated artifacts are ignored by Git.
Only `.gitkeep` placeholders are tracked beneath the data and artifact roots.

## Environment setup

Python 3.13 is the supported runtime.
`pyproject.toml` is the source of truth for direct dependencies, and `uv.lock` records the exact resolved environment.

For development, install [uv](https://docs.astral.sh/uv/) and run:

```powershell
uv sync --frozen
```

To run the graphical labeler on Windows without using uv directly, double-click `Launch_Labeler.bat`.
The launcher creates `.venv`, installs the generated locked dependency export, and opens the Streamlit interface.

`requirements_labeler.txt` is generated from `uv.lock` and must not be edited manually.
After intentionally changing direct dependency constraints, regenerate the lock and compatibility export with:

```powershell
uv lock
uv export --frozen --no-dev --no-hashes --format requirements-txt --output-file requirements_labeler.txt
```

## Audio traversal and spectrogram generation

The generator is `scripts/slice_audio.py`.
It recursively discovers `.wav` and `.mp3` recordings, preserves their relative folder structure, and writes one PNG for each complete chunk.

The current defaults are:

- Input root: `raw/external`
- Output root: `processed/external/spectrograms`
- Sample rate: 22,050 Hz mono
- Chunk duration: five seconds
- Chunk overlap: none
- Mel bins: 128
- Frequency range: 0 Hz to 8,000 Hz
- Final partial chunk: dropped without padding

Each filename records the original recording stem and chunk start time.
For example, `recording01_start30s.png` represents the five-second window beginning at 30 seconds in `recording01.wav`.

Process the default external recordings with:

```powershell
uv run python scripts/slice_audio.py
```

Run a small external-data smoke sample with:

```powershell
uv run python scripts/slice_audio.py --limit-files 2
```

Process pond recordings into their separate tree with:

```powershell
uv run python scripts/slice_audio.py --raw-root raw/ponds --out-root processed/ponds/spectrograms
```

The script logs individual file errors and continues processing the remaining recordings.

## Manual labeling

The recommended interface is `scripts/label_frontend.py`, launched through `Launch_Labeler.bat`.
It displays a spectrogram, locates or exports the matching five-second WAV chunk, and moves the PNG into one of the canonical label directories.

The two label directories are:

- Positive target: `labeled/litoria_aurea`
- Negative example: `labeled/non_target`

The Streamlit interface provides Frog, Background, Skip, Replay, Undo, and session controls.
The Background button writes to the canonical `non_target` filesystem label.

The terminal and Matplotlib alternative is:

```powershell
uv run python scripts/label_spectrograms.py --limit 12 --shuffle
```

Use `--limit 0` only when intentionally starting an unrestricted session.
The labeler defaults to the external spectrogram, raw-audio, and cached-chunk trees.
Pass the corresponding pond paths together when labeling pond data so audio lookup remains deterministic.

## Manifest generation

The manifest builder is `scripts/build_manifest.py`.
It scans `labeled/litoria_aurea` and `labeled/non_target`, parses recording IDs and start times from filenames, and writes `labeled/manifest.csv`.

Generate the manifest with:

```powershell
uv run python scripts/build_manifest.py
```

The CSV columns are:

```text
image_path,label,label_name,recording_id,start_s,split
```

All chunks from the same recording are assigned to the same split to prevent direct recording leakage.
The current splitter is not class-stratified, so its validation and test partitions must not be trusted until Phase 2 of the roadmap is complete.

## Baseline training

The baseline trainer is `scripts/train_baseline.py`.
It resizes spectrograms to small grayscale arrays and trains a balanced logistic-regression classifier as a pipeline sanity check.

Run it after generating a manifest:

```powershell
uv run python scripts/train_baseline.py
```

The script reports train, validation, and test metrics in memory.
It does not save a model and is not the planned production classifier.
Metrics from a partition that lacks either class are not meaningful.

## Stable processing decisions

- Chunking uses fixed five-second, non-overlapping windows.
- Incomplete final chunks are dropped instead of padded.
- Spectrograms are stored as axis-free PNG files for inspection and image-model compatibility.
- Raw-to-processed relative paths are preserved until labeling.
- Labeled images use `litoria_aurea` and `non_target` as canonical class names.
- Manifest splits group by source recording to reduce leakage.
- Raw audio is treated as immutable source material.

## Next milestone

Phase 2 of the roadmap makes evaluation trustworthy before more model development proceeds.
It adds group-aware class balancing, split validation, filename and path tests, and safeguards for duplicate or unparseable examples.
