# Activity Analytics Design

## Summary

Phase 6 turns detection records into calling-activity patterns that site staff can understand, then puts the whole workflow in an application.
It is split into three parts, each designed, built, and verified on its own: activity analytics, the application, and packaging for people who never use a command line.
This document designs the first part.

One command reads a detection record from Phase 5 and writes an analysis folder: a readable report, three figures, and tables that trace every figure value back to individual five-second windows.
Activity is reported as the share of recorded time in which the model detected a call, always beside how much was recorded, so an hour that was never recorded can never look like an hour without frogs.

## Context

The detection record `results/detections/external-resnet18-20261014T114432Z` holds 618 five-minute recordings from 20 January to 5 April 2022, all recorded between 00:00 and 11:59, with a probability for each of 37,080 windows.
Each recording stores the recorder's start time and the UTC offset the recorder states, UTC+11 for every recording here, including the two made after daylight saving ended on 3 April 2022.
The record's own detection rule is a probability at or above the model's validation-selected threshold, 0.794.
Probabilities sit almost entirely near 0 or 1, so the detected share barely moves between thresholds of 0.5 and 0.99.

## Goals

- Aggregate windows by hour of day and by date, keeping recording effort and detected activity as separate numbers.
- Show which hours of the day and which dates were never recorded.
- Draw a time-of-day chart, a 24-hour clock chart, and a seasonal chart by date.
- Show how the totals change across a set of thresholds.
- Let any figure value be reproduced by filtering one table of windows.
- Record exactly which detection record, model, threshold, and time basis produced the analysis.
- Provide a package interface the application can call directly.

## Non-goals

- No comparison between sites or deployments; one analysis covers one detection record, which the detection workflow already ties to one card or deployment.
- No sunrise- or sunset-relative time, which would need site coordinates.
- No user interface; that is the second part.
- No change to how detections are made.

## Decisions

### One fixed time basis

Time of day and date are computed in one fixed UTC offset for the whole analysis.
Each window's time is its recording's start time, converted to UTC with the recording's own stated offset, plus the window's start, then converted to the analysis offset.
The default analysis offset is the offset most recordings state, so the times match the file names; `--utc-offset` overrides it, for example with `+10:00` for Australian Eastern Standard Time.
A fixed offset is deliberate: frogs do not follow daylight saving, and a wall-clock time zone would shift activity by an hour at every change.
A recording without a stated offset is assumed to use the analysis offset, and the report counts such recordings.
A recording without a start time cannot be placed in time; its windows are counted in the report's coverage and left out of every time-based table and figure.

### Effort and activity

The unit is the five-second window.
For each hour of day and each date the analysis reports recorded windows and recorded minutes, the number of recordings contributing, detected windows, and the calling share: detected windows divided by recorded windows.
The calling share of a bin without recorded windows is empty, never zero.
Each window belongs to the bin that contains its start, so a recording that crosses midnight or the top of an hour contributes to both bins.
Failed recordings contribute no windows; the report counts them.

### Thresholds

The analysis threshold defaults to the record's model threshold, and `--threshold` overrides it.
A threshold table repeats the overall totals at 0.5, 0.7, 0.9, 0.95, 0.99, and the analysis threshold, so a reader sees how much the conclusions depend on it.

### Traceability

`windows.csv` lists every window that was placed in time, with its recording path, window start, analysis date and hour, probability, and whether it counts as a detection at the analysis threshold.
Every value in `activity_by_hour.csv`, `activity_by_date.csv`, and the figures is a count or ratio over rows of that table, and a test proves it.
The recording path leads back to `recordings.csv` and `detections.sqlite` in the detection record.

### Reading the record

The analysis opens `detections.sqlite` read-only, through a new `DetectionStore.open_read_only`, which reads the record's identity from the database itself and refuses anything that is not a detection database of the supported schema.
Analysis never changes a detection record.

### Output and replacement

The default output folder is `results/analytics/<detection record folder name>/`; `--output` names another.
Running the analysis again replaces its files, because they are regenerated from the record.
Every file is written beside its target and then swapped in, so a reader never sees half a file.
If a file is open in another program and cannot be replaced, the command says which file and asks the user to close it and run again.
The write-then-swap step, already used by the detection export, moves into a small shared module `frog_classifier.files` so both use one implementation.

## Outputs

| File | Contents |
| --- | --- |
| `report.md` | Coverage, peak hours, the threshold table, the hour and date tables, the figures, and provenance |
| `time_of_day.png` | Calling share by hour of day above recorded hours by hour of day; unrecorded hours are marked |
| `clock.png` | The same calling shares on a 24-hour clock with midnight at the top; unrecorded hours are shaded |
| `by_date.png` | Calling share by date above recorded hours by date; dates without recordings are gaps |
| `activity_by_hour.csv` | `hour`, `recorded_windows`, `recorded_minutes`, `recordings`, `detected_windows`, `calling_share` for hours 0 to 23 |
| `activity_by_date.csv` | `date`, then the same columns, for every date from the first to the last recorded date |
| `windows.csv` | `recording_path`, `window_start_s`, `date`, `hour`, `probability`, `detected` |

Provenance in the report: the detection record folder, its creation time, the model run and checksum, the preprocessing checksum, the analysis threshold and the model threshold, the analysis UTC offset, and the counts of recordings without an offset, without a start time, and failed.

## Architecture

```text
src/frog_classifier/files.py                 write a file beside its target, then swap it in
src/frog_classifier/inference/store.py       adds DetectionStore.open_read_only
src/frog_classifier/inference/export.py      uses frog_classifier.files
src/frog_classifier/analytics/
  __init__.py      package docstring only
  timebase.py      parse and format UTC offsets, choose the analysis offset, place a window in time
  activity.py      read a record and aggregate effort and detections by hour and date, with thresholds
  figures.py       the three figures
  report.py        the Markdown report
  analyze.py       one analysis: read, aggregate, write every output
scripts/analyze_activity.py                  command wrapper
tests/analytics/                             synthetic tests
```

The analytics package imports no PyTorch, so the application can show analyses without loading the model.

## Command

```powershell
uv run python scripts/analyze_activity.py results/detections/external-resnet18-20261014T114432Z
uv run python scripts/analyze_activity.py results/detections/<record> --threshold 0.9 --utc-offset +10:00 --output results/analytics/<name>
```

## Test strategy

All tests build small detection records with `DetectionStore` in temporary folders.

- Time basis: offsets parse and format, including negative and half-hour offsets; malformed offsets are rejected; the most common offset is chosen; windows convert between offsets and cross midnight correctly.
- Aggregation: hour and date bins count effort, recordings, and detections; unrecorded hours and dates have an empty share; a recording crossing midnight splits between dates; recordings without a start time or offset, and failed recordings, are counted and handled as specified; a threshold override changes detections and not effort; the threshold table is ordered and includes the analysis threshold.
- Traceability: every hour and date value equals the corresponding count over `windows.csv` rows.
- Read-only access: analysis leaves the database bytes unchanged; a non-detection file is refused.
- Output: every file is written; a file that cannot be replaced gives the explanatory error and leaves no partial file; figures are valid PNG images.
- Report: coverage, unrecorded hours, the threshold table, and provenance appear.
- Command: default and explicit output folders, overrides, and the error for a folder that is not a record.

## Acceptance criteria

- The full suite and repository checks pass.
- The analysis of the real record reports 618 recordings over 58 dates, hours 00:00 to 11:59 recorded and 12:00 to 23:59 not recorded, and hour shares that match the detection record.
- Every hour and date value in the real analysis is reproduced from `windows.csv`.
- The figures are inspected and show unrecorded hours and dates as unrecorded.
- The documents describe the command and the results.
