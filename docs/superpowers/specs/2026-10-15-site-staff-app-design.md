# Site Staff App Design

## Summary

This is the second of the three parts of Phase 6.
It puts scanning recordings and reading calling activity into a local web application, so site staff never need a command line.
A person points the app at a memory card or folder, or uploads a few recordings, names the deployment, and presses Scan.
When the scan finishes, the app shows the activity charts, says what was and was not recorded, lets the person listen to any detection, and offers the results as one download.
Packaging the app so it installs and starts with a double-click is the third part and is not designed here.

## Context

Phase 5 built `detect_calls`, which scans a folder into a resumable detection record, and part 1 built `analyze_record` and `read_activity`, which turn a record into activity by hour and date.
Both are command-line tools today.
The labeler already uses Streamlit 1.54, so the app uses the same framework and adds no dependency.
A scan of 618 five-minute recordings takes one to two minutes and loads PyTorch; reading and drawing an analysis takes under a second and does not.
The detector was trained and checked on one site, January to April 2022, between midnight and noon, so the app must say so wherever it shows results.

## Goals

- Choose recordings from a folder or memory card, or upload a few files.
- Show how many recordings were found, and their date range, before scanning.
- Scan with progress, a stop button, and continuation after a stop or a closed app.
- Show results only for finished scans.
- Show the summary, the three charts, the threshold table, coverage, warnings, the model version, and the threshold.
- Let the reader change the threshold and the UTC offset.
- Trace any hour or date to its windows, and listen to any window with its spectrogram.
- Download the results as one zip file.
- Warn about recordings whose recorder clock was clearly never set, and leave them out of the time charts.

## Non-goals

- No double-click launcher or installer; that is part 3.
- No comparison between deployments or sites.
- No label editing; the labeler remains the tool for that.
- No pointing a finished deployment at recordings that moved; listening then asks for the original location.
- No hosting for several users; the app listens on localhost only.
- No two scans at once.

## Decisions

### Two processes

The app process runs the user interface and never imports PyTorch, so it starts quickly and stays small.
Each scan runs in its own process, `python -m frog_classifier.app.scan_job`, which loads the model and calls `detect_calls`.
A scan therefore survives a page refresh, and the app can be restarted while it runs.
Closing the console window that started the app also ends the scan, which is what a person closing the app expects; the next scan continues where it stopped.

### Scan status

The scan process writes `scan_status.json` in its detection record folder: its state, the recordings found, already done, processed, and failed, the device, timestamps, and a message.
States are `starting`, `running`, `finished`, `stopped`, and `failed`.
A background thread rewrites the status every two seconds, so a status whose `running` or `starting` state has not been updated for 30 seconds means the process ended without saying so, and the app reports the scan as interrupted.
The status is written beside its target and swapped in; a swap that fails because the app is reading the file at that moment is retried, and the scan never stops because of it.
The scan's own output goes to `scan.log` in the same folder.

The status also says whether the record holds complete results, `results_ready`.
It becomes false when a scan starts processing new recordings and true when a scan finishes.
A scan refused before it starts, for example because the card in the drive is a different one, leaves it as it was, so earlier results stay visible.

### Stopping

The Stop button creates `stop_requested` in the record folder.
The scan process checks for it after every stored recording and also from its status thread, which interrupts the main thread so a long render cannot delay the stop.
Either way the scan ends like Ctrl+C in the command: finished recordings stay, the unfinished one is rolled back, and the status becomes `stopped`.
Starting a scan removes any old stop request first.
Only one scan runs at a time; while one runs, Scan is disabled with the reason.

### Results only for finished scans

The Results view shows charts only for a deployment whose record holds complete results and is not being scanned.
A deployment still scanning shows its progress instead, and a stopped or interrupted one shows how far it got and a button to continue.
A record made by the command line has no status file and is treated as finished.

### Deployments and names

Every scan belongs to a named deployment, for example "IWS076 card 3".
The name is prefilled from the folder name and required for a drive root or an upload.
Its detection record folder is `results/detections/<name>-<model run>`, so the command line's default names read the same way and a later model never collides with an earlier record.
A name keeps letters, digits, spaces, and `-_.()`; anything else becomes `-`, and Windows reserved names are refused.
A name already used for recordings from another folder is refused with the folder it belongs to; the same name and folder continues that deployment, which also picks up recordings added since.
When the app starts a scan it writes `deployment.json` in the record folder with the name, the source folder, the model run, and the creation time, so a deployment stays listed even while its database cannot be read, for example after a scan was cut off mid-write.
The app lists every record folder under `results/detections`, newest first; a record made by the command line has no `deployment.json`, so its name is the folder name without the model suffix and its source folder comes from the database.

### Choosing recordings

A folder is chosen by pasting its path or with a Browse button that opens the system folder dialog, which works because the app runs on the person's own computer.
The app then shows how many `.wav` and `.mp3` recordings it found and the dates their file names give.
Uploaded files are saved under `raw/uploads/<name>/` and scanned from there, because they are source recordings and stay unchanged.
An upload whose file name already exists there with the same bytes is skipped, and one with different bytes is refused, so a recording is never overwritten.
Large collections should be scanned from their folder, not uploaded; the app says so beside the upload box.

### The model

The app uses the newest saved run under `models/`, whose folder name ends with its training time in UTC.
It reads the run's name and threshold from `run.json` without loading PyTorch, and the scan process verifies every checksum before scanning, as the command does.

### Results view

The Results view shows, in order: the summary lines from part 1; warnings; the model run, threshold, and time basis; the notice that the detector is only validated at one site; the threshold and UTC offset controls; the three charts in tabs; the threshold table; a "Look closer" section; and the download button.
The threshold control defaults to the model's threshold and the offset control to the recorders' own offset; changing either recomputes the analysis in memory.
Look closer picks an hour of day or a date that was recorded, lists its windows with time, probability, and whether each counts as a detection, detections first, and plays any of them.
Playing a window loads its recording with the preprocessing contract of the model run, plays the five seconds raised to full loudness so faint calls can be heard, and shows the spectrogram the model saw.
Each placed window now keeps its full start time in the analysis offset, so the app can show it, and `windows.csv` gains a `time` column.
If the recording is no longer at the deployment's folder, for example because the card was removed, the app says where it was expected.
The download is a zip of the part 1 analysis for the current threshold and offset, plus `detections.csv` and `recordings.csv`.

### Recorders with an unset clock

An AudioMoth whose clock was never set stamps its files 1 January 1970, and many recorders reset to 2000.
A start time before 1 January 2005, or more than 14 hours after the detection record was created, cannot be real; 14 hours is the largest UTC offset, so a recording made just before the scan is never refused.
`read_activity` treats such recordings like recordings without a start time: their windows are left out of every time-based value.
Coverage counts them separately, the summary adds a line saying how many and why, the report adds a provenance row, and the app shows a warning.
This changes part 1 for the command line too, so both give the same numbers.

### Figures in memory

The figure functions accept a binary stream as well as a path, so the app draws them without temporary files.

## Architecture

```text
src/frog_classifier/app/
  __init__.py       package docstring only; nothing imports PyTorch except scan_job
  paths.py          the data root and its folders
  models.py         the newest saved run, read without PyTorch
  dialog.py         the system folder dialog
  deployments.py    names, record folders, the deployment list, and recording previews
  scans.py          the status file, starting a scan process, stopping, and the scan state
  scan_job.py       the scan process: run detect_calls and keep the status current
  uploads.py        save uploaded recordings without overwriting any
  listening.py      a window's audio and spectrogram
  export.py         the results zip
classifier_app/app.py                         the Streamlit interface
src/frog_classifier/analytics/activity.py     impossible start times
src/frog_classifier/analytics/report.py       their summary line and provenance row
src/frog_classifier/analytics/figures.py      binary stream targets
tests/app/                                    service, scan process, and interface tests
```

The data root is the repository unless `FROG_CLASSIFIER_ROOT` names another folder with the same layout; tests use that, and packaging may.

## Command

```powershell
uv run streamlit run classifier_app/app.py
```

## Test strategy

- Names: sanitizing, reserved names, the record folder, and refusal of a name used for another folder.
- Deployment list: names without the model suffix, newest first, and the state of each.
- Previews: counts and date ranges, and an empty or missing folder.
- Scan state: running, finished, stopped, failed, and interrupted by a stale status.
- Scan process: a real scan of synthetic recordings with a synthetic saved run ends `finished` with every recording stored; a stop request ends it `stopped` with only whole recordings stored; a refused start ends `failed` with the reason.
- Uploads: saved, identical skipped, different refused, and folder parts of a name ignored.
- Listening: the right five seconds and image, and a missing recording.
- Export: the zip holds every analysis file and both CSV files.
- Impossible start times: 1970 and future dates left out and counted, with the summary line.
- Interface, through Streamlit's test runner: the scan view previews a folder, starts a scan, and shows it finished; the results view shows the summary, warnings, and charts, reacts to the threshold, lists a recorded hour's windows, plays one, and offers the download; a deployment still scanning shows progress instead of charts; the upload box is present.
- The app modules import no PyTorch.

Streamlit's test runner cannot simulate a file upload or a native dialog, so uploads are tested through the upload service and the dialog function is replaced in tests.

## Acceptance criteria

- The full suite and repository checks pass.
- Started from the documented command, the app scans a folder of real recordings, shows progress, can be stopped and continued, and then shows results that match the command-line analysis of the same record.
- The results of `results/detections/external-resnet18-20261014T114432Z` appear in the app with the numbers from part 1.
- Listening to a detection plays the right five seconds.
- The downloaded zip opens and holds the report, charts, and tables.
- The documents describe the app.
