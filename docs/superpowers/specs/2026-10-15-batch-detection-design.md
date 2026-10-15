# Batch Detection Design

## Summary

Phase 5 runs a saved model over whole folders of field recordings and records a frog probability for every five-second window.
One command takes a model run folder and an input folder, renders every recording exactly as the training images were rendered, predicts on the GPU when one is available, and stores the results in one output folder.
The output folder holds a SQLite database, which is the record and makes interrupted runs resumable, plus two spreadsheet-friendly CSV files: the detected windows and every recording with its status.
Every stored row is traceable to its source file, window, recorder clock time, model run, and preprocessing contract.

Nothing in the labeler, the manifest, the training package, or the stored spectrograms changes behavior.

## Context

The saved run `models/resnet18-20261014T114432Z` is verified and has a validation-selected threshold of 0.794.
The 618 recordings under `raw/external` are 48 kHz mono AudioMoth WAV files of five minutes each, named by start time such as `20220123_010000.WAV`, nested two folders deep.
Each WAV header carries the recorder's own statement, for example `Recorded at 01:00:00 23/01/2022 (UTC+11) by AudioMoth 248D9B0260370525`, which proves the file names are local time at UTC+11.
That offset changes when daylight saving ends, so time-of-day analysis in Phase 6 needs it stored per recording.
The `CONFIG.TXT` beside the recordings belongs to a different device and is ignored.

Rendering one recording takes about 0.6 seconds of CPU after the first file, nearly all of it Mel computation, and predicting its 60 windows takes a few hundredths of a second on the GPU.
The machine has 32 logical CPUs.
At one recording at a time, the 618 recordings take about 6 minutes and a year of hourly recordings from one site about 1.5 hours, so the command renders several recordings in parallel.

## Goals

- Detect calls in every complete five-second window of every recording beneath an input folder, recursively.
- Render windows with the exact preprocessing contract the model was trained with, proven pixel for pixel.
- Store a probability for every window, so a later threshold change needs no re-run.
- Record each recording's source path, start time, UTC offset when known, duration, audio checksum, and status.
- Resume an interrupted run without repeating or duplicating finished recordings.
- Record unreadable audio and file names without a timestamp without stopping the batch.
- Identify the exact model and preprocessing contract in every exported row.
- Measure throughput and where the time goes.

## Non-goals

- No analytics, charts, or effort-normalized rates; that is Phase 6.
- No user-facing application; that is Phase 6.
- No change to the threshold rule or to the saved model.
- No site or recorder inference from folder names; the relative path keeps that information for Phase 6.
- No UTC offset for MP3 files, which carry no AudioMoth header.

## Decisions

### SQLite is the record, CSV is the export

The roadmap expected CSV first and SQLite later.
Resumable processing decided it now: one SQLite transaction per recording means a recording is either fully stored or not stored at all, so an interruption at any moment leaves a consistent record and the next run continues from it.
Doing the same with CSV needs commit markers and cleanup of half-written rows.
SQLite ships with Python, so it adds no dependency, and Phase 6 can query it directly.
The CSV files are regenerated from the database at the end of every run for people who open results in a spreadsheet.

### One output folder per input folder and model

The default output folder is `results/detections/<input folder name>-<model run name>/`, holding `detections.sqlite`, `detections.csv`, and `recordings.csv`.
The database records the model run, the model checksum, the threshold, the preprocessing checksum, the window length, and the absolute input folder.
Re-running with the same output folder and a different model, threshold, preprocessing contract, or input folder is refused with a message to choose another output folder, so one database never mixes results.
The output folder must not lie inside the input folder, because recordings are immutable source material.

### The model run supplies the preprocessing contract

The command reads the preprocessing configuration from the run folder's own `preprocessing.toml`, not from `config/`, so a later change to the tracked contract cannot silently change what a saved model sees.
Before predicting, it checks every file checksum in the run's `run.json` and the architecture, exactly as verification does; the model loading moves into one shared function used by both.

### Rendering is shared with the slicer

`frog_classifier.preprocessing` gains `render_waveform_images`, which yields each window's start and 8-bit image, and `render_waveform` becomes that function plus PNG encoding.
Detection uses the images directly, so slicing and detection run the same pixels through the same code.
A test proves the images equal the decoded PNGs, and the real-data acceptance compares detection probabilities with the saved run's predictions for every labeled window.

### Every window is stored

The database holds one row per window with its probability.
A window is a detection when its probability is at or above the model's threshold, which is computed when exporting rather than stored, so Phase 6 can explore other thresholds from the same database.
`detections.csv` lists only detected windows; `recordings.csv` lists every recording with its window and detection counts, which is the recording effort Phase 6 needs.

### Recording time

The start time comes from the `YYYYMMDD_HHMMSS` timestamp in the file name, the convention every earlier phase uses.
When the file name has none, the AudioMoth header time is used instead.
The UTC offset comes from the AudioMoth header, read by walking the WAV file's RIFF chunks to the `LIST` `INFO` `ICMT` comment.
`(UTC)` alone means an offset of `+00:00`.
When both times exist and disagree, the file name wins and the recording's message says so.
A recording with no usable time is still processed; its start time is empty and its message says why.
Times are stored as local ISO 8601 text without an offset, beside the offset as text such as `+11:00`.

### Resume and failure

Recordings are identified by their path relative to the input folder, in POSIX form, and processed in sorted path order.
A recording stored with status `ok` is skipped on the next run, after a check that the file under that name is still the same file: the SHA-256 of its first 64 KiB, stored as `head_sha256`, must match, because recorders on the same schedule write identical file names on every card.
An input folder without a name, such as a drive root, has no default output folder and needs `--output`.
A recording that failed is stored with status `failed` and the error message, and is retried on the next run, replacing its earlier row.
Errors while reading, hashing, or rendering a recording count as that recording's failure; errors in prediction, in the database, or in a worker process stop the run, which can then be resumed.
A recording shorter than one window is stored as `ok` with zero windows and a message.
Recordings already stored stay in the record if they later disappear from the input folder.
Pressing Ctrl+C cancels the recordings not yet started, stores nothing partial, and prints how to continue; workers ignore Ctrl+C so the main process alone decides how to stop.
The command exits with 0 when every recording succeeded and the CSV files were written, 1 when a recording failed or a CSV file could not be replaced, 2 when the run could not start, and 130 after Ctrl+C.

### Parallel rendering

Rendering runs in a pool of worker processes; prediction and database writes stay in the main process, which owns the GPU and the only database connection.
The default is half the logical CPUs, at most eight, and `--workers 1` renders in the main process.
At most twice as many recordings as workers are in flight, so memory stays bounded: each worker holds one decoded recording, about 26 MB for five minutes, and each finished recording waits as about 1.7 MB of images.
Results are consumed in path order, so database writes are deterministic.
Workers are started with the spawn method on every platform, because forking a process that has initialized CUDA is unsafe.
Workers import only the rendering code, not PyTorch; the command imports the detection package inside its main function so that spawned workers, which re-run the command file's top-level imports, stay light.

### Throughput

Each recording row stores its rendering seconds and prediction seconds.
At the end the command prints recordings processed, failures, windows, detections, audio hours, wall time, audio hours per wall-clock hour, and the share of time spent rendering versus predicting.

## Storage

```sql
CREATE TABLE scan (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    input_root TEXT NOT NULL,
    model_run TEXT NOT NULL,
    model_sha256 TEXT NOT NULL,
    threshold REAL NOT NULL,
    preprocessing_sha256 TEXT NOT NULL,
    chunk_seconds INTEGER NOT NULL
);
CREATE TABLE recordings (
    path TEXT PRIMARY KEY,
    recording_id TEXT NOT NULL,
    started_at TEXT,
    utc_offset TEXT,
    size_bytes INTEGER,
    audio_sha256 TEXT,
    head_sha256 TEXT,
    duration_s REAL,
    windows INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ok', 'failed')),
    message TEXT,
    render_seconds REAL NOT NULL,
    predict_seconds REAL NOT NULL,
    processed_at TEXT NOT NULL
);
CREATE TABLE windows (
    path TEXT NOT NULL REFERENCES recordings (path),
    start_s INTEGER NOT NULL,
    probability REAL NOT NULL,
    PRIMARY KEY (path, start_s)
);
```

A database whose schema version differs from the supported one is refused.

Unknown values are stored as NULL; a file that vanished before it could be read has no size.

`detections.csv` columns: `recording_path`, `recording_id`, `window_start_s`, `window_end_s`, `window_start_time`, `utc_offset`, `probability`, `threshold`, `model_run`, `model_sha256`, `preprocessing_sha256`.
Probabilities and the threshold have eight decimals, as in the run's own predictions.

`recordings.csv` columns: `recording_path`, `recording_id`, `started_at`, `utc_offset`, `status`, `windows`, `detections`, `duration_s`, `audio_sha256`, `message`.

Both are written to a temporary file and moved into place.
If a CSV is open in a spreadsheet program and cannot be replaced, the command says so and explains that the database is complete and re-running rewrites the CSV files instantly.

## Architecture

```text
src/frog_classifier/preprocessing/recording.py   adds render_waveform_images
src/frog_classifier/training/saved.py            load a verified run for prediction (shared with verify)
src/frog_classifier/inference/
  __init__.py                package docstring only
  recordings.py              find recordings and read their start time and UTC offset
  rendering.py               render one recording into window images and metadata (no PyTorch)
  store.py                   the SQLite detection store
  export.py                  write detections.csv and recordings.csv
  detect.py                  one detection run: resume, parallel rendering, prediction, storage, summary
scripts/detect_calls.py      command wrapper with a progress bar
tests/inference/             synthetic tests for every module and the command
results/detections/          default output root, ignored by Git
```

## Command

```powershell
uv run python scripts/detect_calls.py models/resnet18-20261014T114432Z raw/external
uv run python scripts/detect_calls.py models/resnet18-20261014T114432Z raw/external --output results/detections/site-a --workers 4 --device cpu
```

## Test strategy

All tests use synthetic audio in temporary folders, a tiny model trained on the CPU without downloads, and `--workers` values of one and two so both rendering paths run.

- Rendering: window images equal the decoded PNGs from `render_waveform`.
- Recordings: discovery is recursive, case-insensitive on suffixes, and sorted; file-name times parse; impossible dates and missing timestamps give no time and a message; a synthetic AudioMoth comment gives the offset; `(UTC)` gives `+00:00`; a disagreeing header gives a message.
- Saved runs: a verified run loads with its threshold and preprocessing copy; a tampered file is refused.
- Store: a new database records its identity; a different identity or schema is refused; a recording and its windows are stored together; a failed recording is replaced on retry; finished paths are reported.
- Detection: every window of every recording is stored; an unreadable file is recorded as failed and the rest finish; a short file has zero windows; an interrupted run resumes without duplicates; two runs on the same input give identical probabilities; detection probabilities equal predictions on the decoded slicer PNGs; an output folder inside the input folder is refused.
- Export: detected windows and recording counts match the database at the threshold; a locked CSV gives the explanatory error.
- Command: arguments reach the package, the default output folder is named from the input and the run, and the summary prints.
- Repository: the inference modules and command are required files, `results/detections/` is a tracked empty root whose contents are ignored, and the workflow documents the command.

## Acceptance criteria

- The full suite, script compilation, `uv lock --check`, `git diff --check`, and the em dash scan pass.
- Detection over all 618 recordings in `raw/external` finishes with every recording stored and the throughput recorded.
- For every labeled window in the saved run's predictions, the detection probability matches the saved probability within 0.0001.
- Interrupting the real run and starting it again finishes without duplicates.
- The README, outline, architecture, workflow, and roadmap describe the command and the measured results.

## Known limitations after this change

- Whole recordings are decoded into memory, as the per-recording noise floor requires; files of many hours need correspondingly more memory per worker.
- MP3 files have no UTC offset.
- The threshold is the saved run's validation-selected value; Phase 6 adds threshold exploration.
