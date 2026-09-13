# Label Audit and Unsure Workflow Design

## Summary

Phase 3 needs more labels, and the labels that exist need one careful re-hearing because the original session had no way to record uncertainty.
This design adds an explicit unsure state, a small append-only decision log, an audit mode in the Streamlit labeler, and a night-first labeling queue, while keeping folders as the only source of truth for training data.
The Matplotlib terminal labeler is retired and its audio export logic moves into a reusable package.
Nothing in the manifest schema, the reports, the sync command, or the verify command changes.

The purpose is to get to a trained model sooner with labels that can be trusted, not to build labeling infrastructure for its own sake.
One person labels, on this machine.

## Goals

- Give uncertainty a home so it is never converted into a negative label again.
- Let the existing 61 labels be re-heard and confirmed, changed, or marked unsure without duplicating or losing files.
- Record every decision so a changed mind leaves a trace.
- Order new clips so the calling hours are reviewed first and no single recording dominates a session.
- Show progress by class and by distinct recording on the labeling screen.
- Keep every existing pipeline command working unchanged.

## Non-goals

- No model-ranked queue, background categories, free-text notes, or multiple reviewers.
- No consistency-measurement tool; the log makes one possible later.
- No change to the manifest schema, folds, reports, sync, verify, or the preprocessing contract.
- No analytics.
- No removal of the now-unused `simpleaudio` dependency, because the lock cannot be regenerated in this environment; it is recorded as a follow-up.

## Current-state findings

A label is the folder a PNG sits in: `labeled/litoria_aurea/` or `labeled/non_target/`.
Skip leaves a clip in the queue with no record, so a later session cannot tell "not reviewed" from "could not decide".
The Streamlit labeler `scripts/label_frontend.py` imports `scripts/label_spectrograms.py` for path handling and audio export, and both display and move files.
Pointing the current labeler at a labeled folder and pressing the same class would create a `__dup` copy, so it cannot audit in place.
The manifest builder rejects any folder under `labeled/` other than the two classes.
Two clips from the original session were listened to and never labeled, and 25 of the 48 background labels come from recordings made between midnight and 6 am, which is where an unsure call could be hiding.

## Architecture

```text
src/frog_classifier/labeling/
  __init__.py
  queue.py          night-first session queue with a per-recording cap
  decisions.py      decision log, file moves, undo, and progress counts
  audio.py          locate or export the five-second playback clip
scripts/
  label_frontend.py Streamlit labeler with label and audit modes (rewritten)
labeled/
  litoria_aurea/    confident target calls (unchanged)
  non_target/       confident background (unchanged)
  unsure/           could not decide; excluded from training
  decisions.csv     append-only decision log
tests/labeling/
  test_queue.py
  test_decisions.py
  test_audio.py
```

`scripts/label_spectrograms.py` is deleted.
`Launch_Labeler.bat` is unchanged and keeps launching the Streamlit labeler.
The `labeling` package depends on `frog_classifier.data` for the naming contract and configuration, on `frog_classifier.preprocessing.display` for colour, and on librosa and the standard library `wave` module for audio export.

### Decision record

`labeled/decisions.csv` has the columns `example_id`, `decision`, `action`, `faint`, and `decided_at`.
`decision` is where the clip is after the action: `litoria_aurea`, `non_target`, `unsure`, or `queue`.
`action` is `label` when a clip leaves the queue, `confirm` when an audited clip keeps its class, `change` when an audited clip moves to another class, and `undo` when the last move of the session is reverted.
`faint` is `1` when the Frog faint button was used and `0` otherwise.
`decided_at` is an ISO 8601 UTC timestamp with second precision.
The file is created with its header on first use, appended one row per button press, and never rewritten.
Nothing reads the log to build training data.

### Folders and the manifest

`labeled/unsure/` is a holding folder.
`discover_labeled_examples` skips the holding folder instead of reporting it as an unknown label directory, so unsure clips never enter the manifest, folds, or reports.
Any other unexpected folder is still rejected.
A tracked `labeled/unsure/.gitkeep` makes the folder exist on a fresh clone.
The decision log is not a PNG and is already ignored by discovery.

### Queue

`build_queue(image_paths, *, seed, per_recording_cap=5, daytime_share=0.1)` returns the session order.
The hour of day is read from the recording ID, which contains a `YYYYMMDD_HHMMSS` timestamp; a recording ID without one counts as daytime.
Night is 19:00 through 06:59.
Night clips are shuffled with the seed, then limited to the cap per recording ID; daytime clips are shuffled and capped the same way.
The result interleaves nine night clips with one daytime clip until both lists are exhausted.
The same function orders the audit queue, so the night-time labels come first there too.

### Moves and undo

`record_decision(image_path, decision, *, labeled_root, faint, log)` moves the clip into `labeled_root / decision / name` and appends a row.
If the clip is already in that folder the function raises, because a confirm is a different action.
If a file with that name already exists in the destination the function raises, so nothing is ever overwritten.
The action is `label` when the clip came from outside the labeled root and `change` otherwise.
`confirm_decision(image_path, *, labeled_root, log)` appends a `confirm` row with the clip's current class and moves nothing.
`undo_move(move, *, log)` moves the clip back to where it came from and appends an `undo` row whose decision is `queue` or the class folder it returned to.
Every move uses an atomic per-file replace onto a destination that does not exist.

### Audio

`find_or_export_chunk(image_path, *, raw_root, chunk_root, config)` returns the playback WAV for a clip.
It parses the example name, locates the recording by stem beneath the raw root (the mirrored folder first when the image sits under a known spectrogram root, then a recursive search requiring exactly one match), and derives the chunk path as `chunk_root / <recording folder relative to raw root> / <example name>.wav`.
If that file exists it is returned; otherwise the five-second window is loaded with librosa at the configured sample rate and written as a mono 16-bit PCM WAV with the standard library.
This is the terminal labeler's logic with one change: the chunk folder always follows the recording's folder, so audited clips under `labeled/` find the same cached WAV the queue produced.

### Streamlit labeler

The sidebar has a mode switch: "Label new clips" or "Audit labeled clips".
Label mode reads the queue root (default `processed/external/spectrograms`); audit mode reads the three labeled folders.
Both modes show the clip through the viridis colour map, the audio player, and the buttons Frog, Frog faint, Background, Unsure, Skip, and Undo.
In audit mode the current class is shown, pressing the same class confirms, and pressing a different class changes it.
Skip leaves no record.
Undo reverts only the last move of the session.
Advanced settings keep the path fields, the seed, the session limit, and the per-recording cap.
The progress row shows counts of frog, background, and unsure clips, distinct recordings per class, and the number remaining in the session.
The screen stays thin: every decision, move, queue, and audio call goes through the package.

### Progress

`summarize_labels(labeled_root)` returns per-class clip counts and distinct recording counts, including unsure.
It is displayed on the labeling screen and has no separate report.

## Test strategy

All tests use synthetic files in temporary directories.

`tests/labeling/test_queue.py`: night clips precede daytime clips at the stated ratio, the cap holds per recording, a recording ID without a timestamp counts as daytime, and the same seed gives the same order.

`tests/labeling/test_decisions.py`: a label moves the file and appends a `label` row, a change appends a `change` row, a confirm appends without moving, moving into the current folder raises, an existing destination file raises and nothing moves, undo restores the file and appends an `undo` row, the log is created with its header once, and `summarize_labels` counts clips and distinct recordings per class.

`tests/labeling/test_audio.py`: a cached chunk is returned without touching raw audio, a missing chunk is exported from a synthetic recording at the right window and sample rate, an image whose recording cannot be found returns nothing, and an ambiguous stem returns nothing.

`tests/data/test_manifest.py`: a PNG under `labeled/unsure/` is skipped and a PNG under any other unexpected folder is still rejected.

`tests/test_repository_reproducibility.py`: the terminal labeler is removed from the operational file list, the labeling modules are required, and `labeled/unsure/.gitkeep` is a tracked placeholder.

The Streamlit screen is checked by hand: label one synthetic clip in each mode and confirm the log rows and file locations.

## Migration and compatibility

1. Implement the package, the manifest rule, the labeler rewrite, and the tests; delete the terminal labeler.
2. Create `labeled/unsure/` in the working data and confirm the manifest still builds with 61 examples.
3. Run the audit on the 25 night-time background clips first, then the 13 positives, then the remaining background clips.
4. Rebuild the manifest after the audit and record the new counts in the roadmap.

No existing label is deleted or renamed by this change.
The two commands that move labeled files, sync and undo, refuse to overwrite.

## Acceptance criteria

- A clip marked Unsure sits in `labeled/unsure/`, appears in the decision log, and never appears in the manifest.
- Auditing a labeled clip and pressing its current class appends a `confirm` row and moves nothing.
- Auditing a labeled clip and pressing another class moves it once and appends a `change` row.
- Undo restores the previous location and appends an `undo` row.
- The queue puts night clips first at nine to one and never exceeds the per-recording cap.
- `scripts/build_manifest.py` succeeds with clips present in `labeled/unsure/`.
- The full suite, script compilation, `git diff --check`, and the em dash scan pass.
- `scripts/label_spectrograms.py` no longer exists and no document refers to it.

## Known limitations after this change

- One reviewer only; the log has no reviewer column.
- Consistency between sessions can be measured from the log but no tool does it yet.
- The hour of day comes from the file name; recordings named without a timestamp are treated as daytime.
- `simpleaudio` remains a declared dependency although nothing imports it; remove it when the lock can be regenerated.
