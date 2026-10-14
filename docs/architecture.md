# Frog Classifier Architecture

This document describes the implemented end-to-end architecture and storage contracts.
The [roadmap](../roadmap.md) records future capabilities and acceptance criteria.

## Pipeline

```mermaid
flowchart LR
  A["Raw recordings<br/>raw/external or raw/ponds"] -->|"slice and create Mel images"| B["Spectrogram queue<br/>processed/*/spectrograms"]
  B -->|"review image and audio"| C["Labeled images<br/>labeled/litoria_aurea or labeled/non_target"]
  B -->|"locate or export exact window"| G["Playback cache<br/>processed/*/chunks"]
  G --> C
  C -->|"validate and assign grouped folds"| D["Versioned manifest<br/>labeled/manifest.csv"]
  D -->|"write reproducibility reports"| R["Data-quality reports<br/>results/data_quality"]
  D -->|"load validated rows"| E["Strict baseline<br/>logistic regression"]
  D -->|"train, rotate folds, choose threshold"| M["Saved run<br/>models/resnet18-time"]
  M -->|"recompute every probability"| V["Verified predictions"]
```

## Storage contract

```text
raw/                         immutable source recordings
processed/                   reproducible spectrograms and playback chunks
labeled/                     human-selected PNGs and generated manifest.csv
labeled/unsure/              clips the reviewer could not decide on, excluded from training
labeled/decisions.csv        append-only log of labeling decisions
config/preprocessing.toml    versioned preprocessing and class contract
config/training.toml         versioned training contract
results/data_quality/        generated manifest validation reports
models/<run>/                saved training runs: weights, input copies, predictions, metrics, report
results/analytics/           future inference and analysis outputs
src/frog_classifier/data/    manifest, splitting, validation, reporting, and configuration package
src/frog_classifier/preprocessing/   Mel rendering, per-recording noise floor, PNG encoding, display colours
src/frog_classifier/labeling/   session queue, decision log, moves, playback audio
src/frog_classifier/training/   training configuration, model, training loop, metrics, saved runs, verification
docs/decisions/              dated design decisions with their evidence
```

Generated recordings, images, manifests, reports, models, and analytics are ignored by Git.
Tracked `.gitkeep` files preserve only the intended empty roots.
The manifest workflow reads labeled images directly, so materialized split directories are not part of the repository layout.

## Preprocessing contract

[`config/preprocessing.toml`](../config/preprocessing.toml) is the machine-readable source of truth for the fixed-window audio, Mel-spectrogram, normalization, rendering, and canonical class settings.
Schema version 2 declares 22,050 Hz mono audio, non-overlapping five-second windows, dropped incomplete final chunks, 128 Mel bins spanning 400 Hz to 4,000 Hz, a per-recording noise floor at the 50th percentile, a 0 dB to 30 dB display range, 8-bit grayscale PNG output with the lowest band at the bottom, and the `litoria_aurea: 1` and `non_target: 0` mapping.
The loader accepts only schema version 2 and tells the operator to regenerate spectrograms when it meets an earlier schema.
The slicer takes every preprocessing value from this file and offers no command-line overrides.
The data package reads the exact TOML bytes and includes their SHA-256 checksum in each generated manifest row.
It rejects non-finite floating-point values before preprocessing begins.

The band was corrected on September 13, 2026 after regeneration from raw audio proved the existing spectrograms were produced with 400 Hz to 4,000 Hz rather than the previously declared 0 Hz to 8,000 Hz.
The noise-floor rendering was adopted the same day; the comparison and reasoning are in [the rendering decision](decisions/2026-09-13-spectrogram-rendering.md).

## Spectrogram rendering

`frog_classifier.preprocessing` renders one recording at a time.
It loads the whole file as mono audio at the configured rate, computes the Mel power spectrogram of the entire file in absolute decibels, and takes the configured percentile of every band across all frames as that band's noise floor.
Each complete five-second chunk is then converted from its own samples, the floor is subtracted per band, and decibels from `db_floor` to `db_ceiling` map linearly to grey levels 0 to 255 with clipping outside that range.
Rows are flipped so the lowest band is the bottom row, and the array is written as a lossless 8-bit grayscale PNG with no metadata, so identical audio always yields identical bytes.
Under the current contract every image is 216 pixels wide and 128 pixels tall.
The labeler shows these images through the viridis colour map for readability; storage stays grayscale.

## Regeneration and verification

`scripts/slice_audio.py` overwrites images by name and never deletes.
`scripts/sync_labeled_images.py` moves each freshly rendered image over its labeled counterpart by name after all names are checked, so a label keeps its identity and the queue no longer holds a labeled example.
`scripts/verify_spectrograms.py` re-renders a seeded sample, or every image with `--all`, from raw audio and compares bytes, which proves the stored images match the tracked contract.

## Human labeling

`scripts/label_frontend.py` is the only labeler; `Launch_Labeler.bat` starts it.
It shows a spectrogram through the viridis colour map with its five-second audio window, which `frog_classifier.labeling.audio` locates in the playback cache or exports from the recording on first use.
In label mode the queue comes from `frog_classifier.labeling.queue`: night clips (19:00 to 06:59) shuffled with a seed, at most five per recording per session, with one daytime clip for every nine night clips.
In audit mode the queue is the labeled folders in the same order without the per-recording cap, the current class is shown, pressing it confirms, and pressing another class moves the clip.
Frog, Frog faint, Background, and Unsure move the PNG into `labeled/litoria_aurea`, `labeled/non_target`, or `labeled/unsure` through `frog_classifier.labeling.decisions`, which refuses to overwrite and appends one row per press to `labeled/decisions.csv` with the resulting folder, the action (`label`, `confirm`, `change`, `undo`, or `return`), the faint flag when a Frog button set it and an empty cell otherwise, and a UTC timestamp.
Skip leaves no record, and Undo steps back through the session's moves one clip per press, newest first, for up to the last 20 moves.
A third mode, Review history, lists labeled clips newest first from the folders and the log, and lets any clip be relabeled or returned to its mirrored place in the queue, which is logged as `return`.
The manifest builder skips `labeled/unsure/`, so unsure clips never enter training.
Changing any path, seed, limit, or cap in the labeler rebuilds its queue, so a decision always applies to the roots shown on screen.

## Versioned manifest and grouped folds

`scripts/build_manifest.py` delegates to `frog_classifier.data` to discover labels, validate filenames and paths, create a class-aware grouped fold plan, and atomically write outputs.
Each row includes `manifest_version`, `example_id`, `image_path`, class metadata, `recording_id`, `start_s`, `fold`, `split`, and `preprocessing_config_sha256`.
All examples from one recording ID receive one fold and one split, preventing recording-level leakage.
The default command uses five folds with fold zero for test data and fold one for validation data.
The validator treats the selected label root as authoritative and requires the first path component beneath it to match the row class.
It re-parses each filename and requires canonical POSIX paths without dot or parent aliases, lowercase PNG suffixes, example IDs, recording IDs, nonnegative aligned start times, and configured class metadata to agree.
It also rejects invalid schemas, missing classes, duplicate identities, missing files, split leakage, and folds or splits that lack either class.
Output safety permits only the lexical, non-symlinked default `labeled/manifest.csv` artifact inside the canonical label tree.
It rejects resolved destination aliases that collide with inputs or fall inside raw, processed, configuration, or selected label source trees.

## Data-quality reports

The manifest command writes JSON and Markdown reports to `results/data_quality/` beside the manifest output.
Each report records manifest and report versions, the fold plan, configuration and manifest checksums, class and recording-group counts, split and fold counts, and validation checks.
The public report builder validates rows, requires the supplied manifest payload to equal their canonical serialization, and verifies exact fold and split agreement with the supplied plan before producing those checks.
The reports make every training and evaluation input inspectable without tracking generated artifacts.

## Strict baseline

`scripts/train_baseline.py` loads the manifest through the same configuration and validation contract.
It trains a balanced logistic-regression classifier on resized grayscale spectrograms only after the strict manifest workflow has produced valid train, validation, and test rows.
The baseline is a data-path and evaluation sanity check rather than a production classifier.
It does not persist a model or perform recording inference.
Its featurization and classifier live in `frog_classifier.training.baseline`, so the ResNet run fits the identical baseline for comparison.

## Model training

`scripts/train_model.py` delegates to `frog_classifier.training.run`, which loads and validates the preprocessing configuration, [`config/training.toml`](../config/training.toml), and the manifest, then snapshots those three inputs and the git commit before anything trains.
The package `__init__` re-exports nothing, so importing the baseline module never loads PyTorch, and nothing outside the package imports it.

| Module | Responsibility |
| --- | --- |
| `config` | Load and validate the training contract and record its checksum |
| `images` | Decode stored PNGs into one array, checking the contract's 128 by 216 shape, and checksum each file |
| `augment` | Seeded training-batch augmentation: circular time shift, attenuation of energy above the noise floor, and one time and one Mel mask |
| `model` | ResNet-18 with ImageNet weights, one-channel input repeated to three ImageNet-normalized channels, and a single frog logit |
| `fit` | AdamW with cosine annealing, early stopping on validation average precision, batch prediction, and full fp32 arithmetic so GPU and CPU predictions agree |
| `metrics` | The threshold rule, split metrics, errors by recording hour, and recall on faint-tagged clips |
| `baseline` | Logistic-regression features and fitting shared with `scripts/train_baseline.py` |
| `artifact` | Staged run folders, checksums, and CSV and JSON reading and writing |
| `report` | The Markdown report and the precision-recall figure, built only from `run.json` content |
| `run` | One complete run from the manifest to a saved folder |
| `verify` | Reproduce a saved run's predictions |

Rotation zero trains on the manifest's training folds, keeps the epoch with the best validation average precision, and is the saved model.
Its threshold is the validation probability with the highest F1, ties going to the higher threshold, and the test fold only reports results.
Cross-validation adds one rotation per remaining fold, each shifting the test and validation folds by the same offset, so every fold is tested once and every labeled clip receives an out-of-fold probability.
The logistic-regression baseline is fitted and thresholded the same way in every rotation.
Faint tags come from the decision log at training time and are stored with the predictions.

A run is written to `models/.resnet18-<UTC time>.partial/` and renamed to `models/resnet18-<UTC time>/` only after `run.json` records the SHA-256 of every other file, so an interrupted run never looks finished and no run overwrites another.
`run.json` also records the architecture, starting weights, input shape, class mapping, threshold and rule, seed, device, library versions, the checksums of the three inputs and `uv.lock`, the git commit and whether the worktree had uncommitted changes, and every metric.

`scripts/verify_model.py` delegates to `frog_classifier.training.verify`, which checks every recorded checksum, reloads the weights without downloading anything, re-reads every image listed in `predictions.csv`, and compares each recomputed probability with the saved one within 0.0001.
An image that is gone or whose bytes changed since training is reported as such rather than as a prediction mismatch.

## Environment contract

Python 3.13 is supported.
`pyproject.toml` declares direct runtime dependencies, `uv.lock` records the resolved environment, and `requirements_labeler.txt` is the generated pip-compatible export used by the Windows launcher.
PyTorch and torchvision form the default `model` dependency group, resolved from the explicit PyTorch CUDA 13.0 index on Windows and Linux; the labeler export leaves the group out.
Create the environment with `uv sync --frozen`.
