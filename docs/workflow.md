# Frog Classifier Workflow

This is the operational guide for preprocessing recordings, applying human labels, and producing a validated manifest.
Read the [README](../README.md) for the project entry point and the [architecture](architecture.md) for implementation boundaries.

## Environment setup

Python 3.13 is the supported runtime.
Install [uv](https://docs.astral.sh/uv/) and create the locked environment:

```powershell
uv sync --frozen
```

The sync also installs the default `model` dependency group, which holds PyTorch and torchvision.
On Windows and Linux they come from the PyTorch CUDA 13.0 package index, so an NVIDIA GPU with a recent driver is used when present and the CPU otherwise.

To open the graphical labeler on Windows without using uv directly, double-click `Launch_Labeler.bat`.
The launcher creates `.venv`, installs the generated locked compatibility export, and opens the Streamlit interface.
`requirements_labeler.txt` is generated from `uv.lock` and must not be edited manually.
It leaves out the `model` group, so the labeler never downloads PyTorch.
After intentionally changing direct dependency constraints, refresh the lock and compatibility export:

```powershell
uv lock
uv export --frozen --no-default-groups --no-hashes --format requirements-txt --output-file requirements_labeler.txt
```

## Generate spectrograms

`scripts/slice_audio.py` recursively discovers `.wav` and `.mp3` recordings, preserves their relative folder structure, and writes one 8-bit grayscale PNG for every complete five-second window.
The configured contract uses 22,050 Hz mono audio, no overlap, 128 Mel bins, a 400 Hz to 4,000 Hz range, and drops the final partial window.
Each image shows decibels above the recording's own per-band noise floor, taken at the 50th percentile over the whole file, with 0 dB to 30 dB mapped to black through white.
The *Litoria aurea* call energy in the labeled examples sits between roughly 500 Hz and 2,500 Hz, so this band keeps the Mel resolution where the call lives.
The machine-readable configuration is [config/preprocessing.toml](../config/preprocessing.toml), and the command offers no overrides for it.
The configuration loader rejects non-finite floating-point values and any schema other than version 2.
The command overwrites existing images by name and deletes nothing.

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

## Replace labeled images after regeneration

Regenerating the queue also recreates every labeled example under its original name.
Run the sync command to move each fresh image over its labeled counterpart:

```powershell
uv run python scripts/sync_labeled_images.py
```

The command checks every labeled name first and changes nothing if any name fails to parse, has no fresh counterpart, or has more than one.
Pass `--dry-run` to see how many images would be replaced.
If the command stops partway, for example because a labeler still holds an image open, it reports how many images were replaced and which one failed.
Run `scripts/slice_audio.py` again to recreate the queue copies that were already consumed, then run the sync again.
The labeled tree is shared by every source, so regenerate every source's queue before syncing, or the command reports missing counterparts for the labels of a source that has not been regenerated.

## Verify stored spectrograms

Prove that stored images match the raw audio and the tracked contract:

```powershell
uv run python scripts/verify_spectrograms.py --sample 50
```

The command draws a seeded sample from the labeled tree and from the queue, re-renders each recording, and prints `MATCH` or the reason for a mismatch per image.
A stored image whose bytes differ but whose pixels match a fresh render is reported as `ENCODING_DIFFERS`, which points at a PNG encoder change rather than a rendering change.
It exits non-zero on any mismatch.
Pass `--all` to check every image, which renders every recording once.

## Apply labels

Double-click `Launch_Labeler.bat`, or run the Streamlit labeler directly:

```powershell
uv run python -m streamlit run scripts/label_frontend.py
```

The sidebar switches between three modes.
"Label new clips" walks the queue under the spectrogram root: night clips first, shuffled with the seed, at most five per recording per session, with one daytime clip for every nine night clips.
"Audit labeled clips" walks the three labeled folders in the same order, shows the current label, and lets you confirm it by pressing the same button or change it by pressing another.
Both modes play the five-second audio window, exporting it from the recording on first use.
"Review history" shows the labeled clips as a gallery, newest decision first, with a filter by label and a Review button on every clip.
Reviewing a clip plays its audio and lets you confirm it, move it to another label, or send it back to the queue so it is unlabeled again.
Advanced settings hold the four roots, the seed, the session limit, and the per-recording cap, and changing any of them rebuilds the queue.
The per-recording cap applies to label mode only, so an audit session re-hears every labeled clip.

Buttons: Frog, Frog faint, Background, Unsure, Skip, and Undo.
Skip leaves no record.
Undo steps back through the session's moves one clip per press, newest first, for up to the last 20 moves.
Every other press appends one row to `labeled/decisions.csv`.

### Labeling policy

Use Frog only when the target call is confidently present, and Frog faint when it is distant.
Use Background only when the clip is confidently non-target.
Use Unsure when identification is uncertain, and never convert uncertainty into a negative label.

Frog is stored as `labeled/litoria_aurea` with numeric label `1`.
Background is stored as `labeled/non_target` with numeric label `0`.
Unsure is stored as `labeled/unsure` and is excluded from the manifest.

### Audit the existing labels

Run audit mode once before training and review the night-time background clips first, since they are where an unsure call could be hiding.
Rebuild the manifest afterwards.

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

## Train and verify the detector

`scripts/train_model.py` fine-tunes an ImageNet-pretrained ResNet-18 on the validated manifest and saves the result as one run folder under `models/`.
The training settings live in [config/training.toml](../config/training.toml), which is validated like the preprocessing configuration and recorded by checksum with every run.
Rebuild the manifest first so training sees the current labels.

```powershell
uv run python scripts/build_manifest.py
uv run python scripts/train_model.py
```

The saved model trains on the manifest's training folds, keeps the epoch with the best validation average precision, and chooses its detection threshold on the validation fold as the probability with the highest F1, ties going to the higher threshold.
The test fold is used only to report results.
By default the command also trains four more models, rotating the folds so every fold is the test fold once, to show how much performance varies between groups of recordings.
Those rotations also give every labeled clip a probability from a model that never trained on its recording, which is the place to look for labels worth re-checking.
Skip the extra rotations for a quick check:

```powershell
uv run python scripts/train_model.py --no-cross-validation
```

A run on the RTX 4080 laptop GPU takes a few minutes; a CPU-only machine works but is much slower.
The first run downloads the ResNet-18 ImageNet weights, about 45 MB, into the PyTorch cache.

Each run folder is named `resnet18-<UTC time>` and holds:

| File | Contents |
| --- | --- |
| `model.pt` | The saved model weights |
| `run.json` | Metadata, threshold, metrics, provenance, and the SHA-256 of every other file |
| `report.md` | Human-readable results |
| `preprocessing.toml`, `training.toml`, `manifest.csv` | Exact copies of the inputs as they were when training started |
| `predictions.csv` | Every manifest clip with its image checksum, faint tag, ResNet probability, and baseline probability |
| `history.csv` | Loss and validation average precision per epoch |
| `cross_validation.csv`, `out_of_fold_predictions.csv` | Per-rotation results and out-of-fold probabilities, when cross-validation ran |
| `precision_recall.png` | Validation and test precision-recall curves with the chosen thresholds |

The report compares the ResNet with the logistic-regression baseline on identical data, states the expected false detections per hour of recording without a frog call, breaks test errors down by recording hour, and gives recall on faint-tagged frog clips separately.
A run is written under a hidden temporary name and renamed only when complete, and the command never reuses an existing folder.

Prove that a saved run reproduces its own predictions:

```powershell
uv run python scripts/verify_model.py models/<run folder>
uv run python scripts/verify_model.py models/<run folder> --device cpu
```

Verification checks every file checksum in `run.json`, reloads the weights without downloading anything, re-reads every image, and recomputes every probability.
A difference above 0.0001 fails, and an image that was relabeled, returned to the queue, or regenerated since training is reported as missing or changed.
The command exits non-zero on any failure.

## Detect calls in recordings

`scripts/detect_calls.py` runs a saved model over every recording beneath a folder and records the frog probability of every five-second window.

```powershell
uv run python scripts/detect_calls.py models/<run folder> raw/external
```

The command finds every `.wav` and `.mp3` file in the folder and its subfolders, renders each recording with the preprocessing contract saved in the run folder, and predicts on the GPU when one is available.
It checks every file checksum in the run's `run.json` before it starts and never writes inside the input folder.
Results go to `results/detections/<input folder>-<run folder>/` unless `--output` names another folder:

| File | Contents |
| --- | --- |
| `detections.sqlite` | The complete record: the model and preprocessing checksums, every recording with its status, and every window's probability |
| `detections.csv` | Every window at or above the run's threshold, with its recording, clock time, probability, and model provenance |
| `recordings.csv` | Every recording with its start time, UTC offset, status, window and detection counts, duration, audio checksum, and any message |

A recording's start time comes from the `YYYYMMDD_HHMMSS` timestamp in its file name, and its UTC offset from the AudioMoth header when the file has one.
A file that cannot be read is recorded as failed, the batch continues, and the command exits with 1.
Running the same command again skips every recording already stored, retries failed ones, and rewrites both CSV files, so an interrupted run simply continues.
An output folder belongs to one input folder, model, and preprocessing contract; the command refuses to mix results and asks for another `--output` folder instead.
A CSV file open in a spreadsheet program cannot be replaced; close it and run the command again.

Recordings are rendered by several worker processes, by default half the logical CPUs and at most eight:

```powershell
uv run python scripts/detect_calls.py models/<run folder> raw/external --workers 4 --device cpu
```

On the RTX 4080 laptop with 32 logical CPUs, the 618 five-minute recordings in `raw/external`, 51.5 hours of audio, took 427 seconds with one worker, 126 seconds with four, 76 seconds with eight, and 94 seconds with sixteen, so the default of eight was the fastest.
Rendering takes nearly all of that time; prediction took about 14 seconds in total.
Every worker count stored exactly the same probabilities, and for all 4,224 labeled clips the detection probability matched the saved run's own prediction within 0.000004.
Each worker process uses about 250 MB of memory for five-minute recordings.
If the command is killed outright, its workers exit on their own, and the next run continues from the last complete recording.

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
uv run python -B -m py_compile scripts/build_manifest.py scripts/label_frontend.py scripts/slice_audio.py scripts/sync_labeled_images.py scripts/train_baseline.py scripts/train_model.py scripts/verify_model.py scripts/verify_spectrograms.py
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
