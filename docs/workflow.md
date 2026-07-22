# Frog Classifier Workflow

This is the operational guide for preprocessing recordings, applying human labels, and producing a validated manifest.
Read the [README](../README.md) for the project entry point and the [architecture](architecture.md) for implementation boundaries.

## Environment setup

Python 3.13 is the supported runtime.
Install [uv](https://docs.astral.sh/uv/) and create the locked environment:

```powershell
uv sync --frozen
```

To open the graphical labeler on Windows without using uv directly, double-click `Launch_Labeler.bat`.
The launcher creates `.venv`, installs the generated locked compatibility export, and opens the Streamlit interface.
`requirements_labeler.txt` is generated from `uv.lock` and must not be edited manually.
After intentionally changing direct dependency constraints, refresh the lock and compatibility export:

```powershell
uv lock
uv export --frozen --no-dev --no-hashes --format requirements-txt --output-file requirements_labeler.txt
```

## Generate spectrograms

`scripts/slice_audio.py` recursively discovers `.wav` and `.mp3` recordings, preserves their relative folder structure, and writes one axis-free PNG for every complete five-second window.
The configured contract uses 22,050 Hz mono audio, no overlap, 128 Mel bins, a 0 Hz to 8,000 Hz range, and drops the final partial window.
The machine-readable configuration is [config/preprocessing.toml](../config/preprocessing.toml).
The configuration loader rejects non-finite floating-point values before they can reach librosa or Matplotlib.
The slicer passes the tracked power, FFT, hop length, figure dimensions, DPI, and interpolation values explicitly to those libraries.

Process the default external recording root:

```powershell
uv run python scripts/slice_audio.py
```

Run a small external-data smoke sample:

```powershell
uv run python scripts/slice_audio.py --limit-files 2
```

Process pond recordings into their separate tree:

```powershell
uv run python scripts/slice_audio.py --raw-root raw/ponds --out-root processed/ponds/spectrograms
```

Each filename records the original recording stem and chunk start time.
For example, `recording01_start30s.png` represents the five-second window beginning at 30 seconds in `recording01.wav`.

## Apply labels

The recommended interface is `scripts/label_frontend.py`, launched through `Launch_Labeler.bat`.
It displays a spectrogram, locates or exports the matching five-second WAV chunk, and moves the PNG into one canonical label directory.
The terminal and Matplotlib alternative is:

```powershell
uv run python scripts/label_spectrograms.py --limit 12 --shuffle
```

Use `--limit 0` only when intentionally starting an unrestricted session.
The labeler defaults to the external spectrogram, raw-audio, and cached-chunk trees.
Pass the corresponding pond paths together when labeling pond data so audio lookup remains deterministic.

### Labeling policy

Use Frog only when the target call is confidently present, even if it is faint.
Use Background only when the clip is confidently non-target.
Use Skip when identification is uncertain, and do not convert uncertainty into a negative label.
Phase 3 will add an explicit review-later state and signal-quality metadata.

Frog is stored as `labeled/litoria_aurea` with numeric label `1`.
Background is stored as `labeled/non_target` with numeric label `0`.

## Build the validated manifest

`scripts/build_manifest.py` discovers the canonical label directories, parses each example name, validates duplicates and paths, and creates class-aware grouped folds.
All examples from one recording ID receive one fold and one split.
The command writes the versioned CSV to `labeled/manifest.csv` and JSON and Markdown data-quality reports to `results/data_quality/`.

```powershell
uv run python scripts/build_manifest.py
```

The manifest includes its version, a stable example identifier, image and class metadata, recording identity, start time, fold, split, and preprocessing configuration SHA-256.
Strict loading treats the selected label root as authoritative and requires the first path component beneath it to match the row class.
It re-parses every image filename and requires a canonical POSIX path without dot or parent aliases, lowercase PNG suffix, example ID, recording ID, nonnegative start time, and configured chunk alignment to agree.
The default plan uses five folds with test fold zero and validation fold one.
Override the fold plan only when the resulting folds preserve both classes and recording isolation:

```powershell
uv run python scripts/build_manifest.py --folds 5 --test-fold 0 --val-fold 1 --seed 1337
```

Only the lexical, non-symlinked default `labeled/manifest.csv` destination remains allowed inside the selected label tree.
Custom manifest destinations must have a CSV suffix, and no output may target the selected configuration, a discovered label image, a canonical label source tree, or the repository raw, processed, or configuration trees.
The command fails instead of writing partial output when the source labels, filename contract, class coverage, fold plan, or output destinations are invalid.
The report builder independently validates every row, requires the supplied manifest bytes to match their canonical serialization, and verifies each fold and split against the supplied split plan before emitting passed checks.

## Run the strict baseline

The baseline loads and revalidates the generated manifest with the same preprocessing configuration.
It trains a balanced logistic-regression classifier on resized grayscale spectrograms and prints train, validation, and test metrics.

```powershell
uv run python scripts/train_baseline.py
```

The baseline is a pipeline and evaluation sanity check.
It does not save a model or perform recording inference.

## Verify Phase 2 acceptance

Run the real-data manifest twice and compare all three generated artifacts:

```powershell
uv run python scripts/build_manifest.py
Get-Content -LiteralPath results/data_quality/manifest-report.md
Get-FileHash -Algorithm SHA256 -LiteralPath labeled/manifest.csv, results/data_quality/manifest-report.json, results/data_quality/manifest-report.md
uv run python scripts/build_manifest.py
Get-FileHash -Algorithm SHA256 -LiteralPath labeled/manifest.csv, results/data_quality/manifest-report.json, results/data_quality/manifest-report.md
```

The verified output paths are:

- `labeled/manifest.csv`
- `results/data_quality/manifest-report.json`
- `results/data_quality/manifest-report.md`

The verified run used seed `1337` with five folds: fold 0 was test, fold 1 was validation, and folds 2 through 4 were training.
The July 22, 2026 acceptance run produced 61 examples from 60 recording groups and identical SHA-256 values across both generations.
Every fold and split contained both canonical classes.

Run the strict baseline and the complete repository verification:

```powershell
uv run python scripts/train_baseline.py
uv lock --check
uv sync --frozen
uv pip check
uv run python -m unittest discover -s tests -v
uv run python -B -m py_compile scripts/build_manifest.py scripts/label_frontend.py scripts/label_spectrograms.py scripts/slice_audio.py scripts/train_baseline.py
git diff --check
```

Scan every changed Markdown and text file for the prohibited em dash character, then inspect the intended tracked changes:

```powershell
$changedText = @(git diff --name-only -- '*.md' '*.txt')
$emDashHits = @()
foreach ($path in $changedText) {
    $emDashHits += @(Select-String -LiteralPath $path -Pattern ([char]0x2014))
}
if ($emDashHits.Count -gt 0) {
    $emDashHits | Format-Table Path,LineNumber,Line
    exit 1
}
Write-Output ("Changed text files scanned: " + ($changedText -join ', '))
Write-Output 'Em-dash matches: 0'
git status --short
```

The verified acceptance run passed 83 tests with zero failures, compiled every operational script, and reported compatible locked dependencies.
