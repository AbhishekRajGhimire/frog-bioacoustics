# Site Staff App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local Streamlit app in which site staff scan a folder, memory card, or uploaded recordings, follow and stop the scan, and read, check by ear, and download the calling activity.

**Architecture:** A torch-free service package `frog_classifier.app` holds deployments, the scan status file, uploads, listening, and export; each scan runs in its own process, `frog_classifier.app.scan_job`, which calls `detect_calls` and keeps `scan_status.json` current. `classifier_app/app.py` is a thin Streamlit interface over those services and the part 1 analytics, which gain impossible-date handling and a window start time.

**Tech Stack:** Python 3.13, Streamlit 1.54 and its `AppTest` runner, matplotlib, SQLite, unittest.

**Spec:** `docs/superpowers/specs/2026-10-15-site-staff-app-design.md`

## Global Constraints

- No new dependencies: `pyproject.toml`, `uv.lock`, and `requirements_labeler.txt` do not change.
- uv is at `/c/Users/ghimi/.local/bin/uv.exe`; run Python as `.venv/Scripts/python.exe`.
- Text files use LF line endings; write files with the editor tools or bytes.
- Never use the em dash character anywhere; each full sentence in long Markdown files sits on its own line.
- Commit messages carry no co-author trailer.
- Tests use synthetic data in temporary folders and never read `raw/`, `labeled/`, `models/`, or `results/`.
- Only `frog_classifier.app.scan_job` may import PyTorch, and only inside its scan function.
- The app listens on localhost only; it never writes inside a folder of source recordings.
- Never kill `python.exe` by name: other projects' servers run on this machine.

## Review Focus

- The app is refreshed or restarted while a scan runs: the scan carries on and the app finds it again from its status (Tasks 4 and 9 tests).
- A memory card is removed after its scan: listening says where the recording was expected instead of failing (Task 7 test).
- The status file is read at the moment the scan swaps in a new one: the read is retried and never mistaken for "no scan" (Task 3 test).
- A rescan refused before it starts, for example because a different card is in the drive, keeps the earlier results visible (Task 5 test).
- The same name chosen for recordings from a different folder is refused with that folder named (Task 4 test).

---

### Task 1: Impossible start times, window start times, coverage warnings, and figures in memory

**Files:**
- Modify: `src/frog_classifier/analytics/activity.py`, `report.py`, `figures.py`, `analyze.py`
- Modify: `tests/analytics/helpers.py`
- Test: `tests/analytics/test_activity.py`, `test_report.py`, `test_figures.py`, `test_analyze.py`

**Interfaces:**
- Produces: `EARLIEST_PLAUSIBLE_START`; `PlacedWindow(recording_path, window_start_s, start: datetime, probability)` with properties `date` and `hour`; `Coverage.recordings_with_impossible_time` after `recordings_without_time`; `headline_lines(activity) -> list[str]`, `coverage_warnings(activity) -> list[str]`, and `summary_lines` returning both; figure writers accept `Path | BinaryIO`; `windows.csv` gains `time` after `date`; `build_record(..., threshold=0.5, **identity_changes)`.

- [ ] **Step 1: Write the failing tests**

In `tests/analytics/helpers.py`, `build_record` gains identity changes:

```python
def build_record(
    record_dir: Path,
    recordings: Sequence[tuple[RecordingRow, tuple[float, ...]]],
    *,
    threshold: float = 0.5,
    **identity_changes: object,
) -> Path:
    """A detection record folder whose model threshold is threshold."""
    record_dir.mkdir(parents=True, exist_ok=True)
    identity = replace(SCAN_IDENTITY, threshold=threshold, **identity_changes)
```

In `tests/analytics/test_activity.py`, add `recordings_with_impossible_time=0,` after `recordings_without_time=1,` in the expected `Coverage`, add to `test_a_recording_that_crosses_midnight_counts_on_both_dates`:

```python
        self.assertEqual(
            [f"{window.start:%Y-%m-%d %H:%M:%S}" for window in activity.windows],
            ["2022-01-23 23:59:55", "2022-01-24 00:00:00"],
        )
```

and add:

```python
    def test_recordings_whose_clock_was_never_set_are_left_out_and_counted(self) -> None:
        # The record was created at 2026-10-15T00:00:00Z.
        activity = read_test_activity(self.record, [
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.9, 0.1)),
            recording("b/19700101_000000.WAV", "1970-01-01T00:00:00", (0.9, 0.9)),
            recording("c/20261016_000000.WAV", "2026-10-16T00:00:00", (0.9,)),
            # Within 14 hours of the scan, so a recorder far ahead of UTC can have made it.
            recording("d/20261015_130000.WAV", "2026-10-15T13:00:00", (0.1,)),
        ])

        self.assertEqual(activity.coverage.recordings_with_impossible_time, 2)
        self.assertEqual(activity.coverage.unplaced_windows, 3)
        self.assertEqual(
            {window.recording_path for window in activity.windows},
            {"a/20220123_010000.WAV", "d/20261015_130000.WAV"},
        )
```

In `tests/analytics/test_report.py`, import `coverage_warnings`, replace the expected summary in `test_a_record_with_nothing_placed_in_time_says_so` with:

```python
            [
                "No recording could be placed in time, so there is no activity to report.",
                "Left out of every result by time: 1 recording with no start time in the file name.",
            ],
```

and add:

```python
    def test_coverage_problems_are_warned_about(self) -> None:
        activity = read_test_activity(self.root / "record", [
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.9,)),
            recording("a/19700101_000000.WAV", "1970-01-01T00:00:00", (0.9,)),
            recording("a/19700101_000500.WAV", "1970-01-01T00:05:00", (0.9,)),
            recording("b/20220123_010000.WAV", "2022-01-23T01:00:00", (0.1,), utc_offset=None),
            recording("a/broken.wav", None, (), utc_offset=None, status="failed", message="LibsndfileError: bad"),
        ])

        self.assertEqual(coverage_warnings(activity), [
            "Left out of every result by time: 2 recordings with an impossible date, before 2005 or after the scan, "
            "probably because the recorder's clock was not set.",
            "Not included because reading failed: 1 recording. Scanning again retries failed recordings.",
            "Taken to use UTC+11:00 because no UTC offset is stated: 1 recording.",
        ])
        self.assertIn("| Recordings with an impossible start time, left out | 2 |", render_report(activity, "site-a"))
```

In `tests/analytics/test_figures.py`, add `import io` and:

```python
    def test_figures_can_be_drawn_into_memory(self) -> None:
        activity = read_test_activity(self.root / "record", [
            recording("a/20220120_010000.WAV", "2022-01-20T01:00:00", (0.9, 0.1)),
        ])

        for writer in WRITERS:
            with self.subTest(figure=writer.__name__):
                buffer = io.BytesIO()
                writer(buffer, activity)
                buffer.seek(0)
                with Image.open(buffer) as image:
                    self.assertEqual(image.format, "PNG")
```

In `tests/analytics/test_analyze.py`, add to the traceability test after `self.assertEqual(len(windows), 8)`:

```python
        self.assertEqual((windows[0]["date"], windows[0]["time"], windows[0]["hour"]), ("2022-01-20", "01:00:00", "1"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.analytics.test_activity tests.analytics.test_report tests.analytics.test_figures tests.analytics.test_analyze`
Expected: errors for the missing `coverage_warnings`, `recordings_with_impossible_time`, `start`, and `time` column; the in-memory figure test already passes, because `savefig` accepts a stream and only the declared interface changes.

- [ ] **Step 3: Implement**

`src/frog_classifier/analytics/activity.py`: import `datetime` beside `date` and `timedelta`, add after `TABLE_THRESHOLDS`:

```python
# A recorder whose clock was never set stamps 1970 (AudioMoth) or 2000 (many
# others), so no real recording here starts before this.
EARLIEST_PLAUSIBLE_START = datetime(2005, 1, 1)
# Start times are a recorder's local clock, which runs at most 14 hours, the
# largest UTC offset, ahead of the record's creation time in UTC.
_LARGEST_CLOCK_LEAD = timedelta(hours=14)
```

Replace `PlacedWindow` with:

```python
@dataclass(frozen=True)
class PlacedWindow:
    recording_path: str
    window_start_s: int
    start: datetime
    probability: float

    @property
    def date(self) -> date:
        return self.start.date()

    @property
    def hour(self) -> int:
        return self.start.hour
```

Add `recordings_with_impossible_time: int` to `Coverage` after `recordings_without_time`.
In `read_activity`, replace the `timed` line with:

```python
    dated = [row for row in succeeded if row.started_at is not None]
    latest_start = datetime.fromisoformat(store.created_at).replace(tzinfo=None) + _LARGEST_CLOCK_LEAD
    timed = {
        row.path: row
        for row in dated
        if EARLIEST_PLAUSIBLE_START <= datetime.fromisoformat(row.started_at) <= latest_start
    }
```

append windows as `PlacedWindow(path, start_s, moment, probability)`, and set the coverage counts:

```python
            recordings_without_time=len(succeeded) - len(dated),
            recordings_with_impossible_time=len(dated) - len(timed),
```

`src/frog_classifier/analytics/report.py`: rename `summary_lines` to `headline_lines` with the docstring `"""The headline facts: what was recorded, how much calling it held, and when."""`, and add:

```python
def coverage_warnings(activity: Activity) -> list[str]:
    """What a reader must know about the recordings before trusting the results."""
    coverage = activity.coverage
    warnings = []
    if coverage.recordings_with_impossible_time:
        warnings.append(
            f"Left out of every result by time: {_recordings(coverage.recordings_with_impossible_time)} "
            "with an impossible date, before 2005 or after the scan, probably because the recorder's clock was not set."
        )
    if coverage.recordings_without_time:
        warnings.append(
            f"Left out of every result by time: {_recordings(coverage.recordings_without_time)} "
            "with no start time in the file name."
        )
    if coverage.failed_recordings:
        warnings.append(
            f"Not included because reading failed: {_recordings(coverage.failed_recordings)}. "
            "Scanning again retries failed recordings."
        )
    if coverage.recordings_without_offset:
        warnings.append(
            f"Taken to use {activity.time_basis.label} because no UTC offset is stated: "
            f"{_recordings(coverage.recordings_without_offset)}."
        )
    return warnings


def summary_lines(activity: Activity) -> list[str]:
    """The headline facts and then the coverage warnings, for the report and the command's output."""
    return [*headline_lines(activity), *coverage_warnings(activity)]


def _recordings(count: int) -> str:
    return "1 recording" if count == 1 else f"{count:,} recordings"
```

Add the provenance row after the row for recordings without a start time:

```python
        f"| Recordings with an impossible start time, left out | {coverage.recordings_with_impossible_time} |",
```

`src/frog_classifier/analytics/figures.py`: import `BinaryIO` from `typing`, rename each writer's `path: Path` parameter to `target: Path | BinaryIO`, and pass `target` to `savefig`.

`src/frog_classifier/analytics/analyze.py`: `_WINDOW_COLUMNS` becomes `("recording_path", "window_start_s", "date", "time", "hour", "probability", "detected")`, and each row gains `"time": f"{window.start:%H:%M:%S}",` after `"date"`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests/analytics -t .`
Expected: every analytics test passes.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/analytics tests/analytics
git commit -m "feat: leave impossible recording dates out of activity and warn about them"
```

---

### Task 2: The app's data root and model

**Files:**
- Create: `src/frog_classifier/app/__init__.py`, `paths.py`, `models.py`
- Create: `tests/app/__init__.py` (empty)
- Test: `tests/app/test_paths_and_models.py`

**Interfaces:**
- Produces: `ROOT_VARIABLE = "FROG_CLASSIFIER_ROOT"`, `AppPaths(root)` with properties `models`, `uploads`, `detections`; `app_paths(default_root) -> AppPaths`; `ModelError(ValueError)`, `ModelInfo(run_dir, name, threshold)`, `newest_model(models_root) -> ModelInfo`.

- [ ] **Step 1: Write the failing tests**

Create `tests/app/test_paths_and_models.py`:

```python
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frog_classifier.app.models import ModelError, ModelInfo, newest_model
from frog_classifier.app.paths import ROOT_VARIABLE, app_paths
from frog_classifier.training.artifact import RUN_METADATA


class AppPathsTests(unittest.TestCase):
    def test_the_root_variable_names_the_data_root(self) -> None:
        with patch.dict(os.environ, {ROOT_VARIABLE: "D:/frog-data"}):
            paths = app_paths(Path("ignored"))

        root = Path("D:/frog-data")
        self.assertEqual(
            (paths.models, paths.uploads, paths.detections),
            (root / "models", root / "raw" / "uploads", root / "results" / "detections"),
        )

    def test_without_the_variable_the_default_root_is_used(self) -> None:
        environment = {name: value for name, value in os.environ.items() if name != ROOT_VARIABLE}
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(app_paths(Path("repo")).root, Path("repo"))


class NewestModelTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _run(self, name: str, threshold: float) -> Path:
        run_dir = self.root / name
        run_dir.mkdir()
        (run_dir / RUN_METADATA).write_bytes(json.dumps({"run_name": name, "threshold": {"value": threshold}}).encode())
        return run_dir

    def test_the_most_recently_trained_run_is_chosen(self) -> None:
        self._run("resnet18-20261014T114432Z", 0.7)
        newest = self._run("resnet34-20261101T000000Z", 0.8)
        self._run("resnet50-20250101T000000Z", 0.9)

        self.assertEqual(newest_model(self.root), ModelInfo(newest, "resnet34-20261101T000000Z", 0.8))

    def test_unfinished_and_unreadable_runs_are_ignored(self) -> None:
        good = self._run("resnet18-20261014T114432Z", 0.7)
        (self.root / ".resnet18-20261201T000000Z.partial").mkdir()
        broken = self.root / "resnet18-20261202T000000Z"
        broken.mkdir()
        (broken / RUN_METADATA).write_bytes(b"{not json")

        self.assertEqual(newest_model(self.root).run_dir, good)

    def test_a_missing_model_is_explained(self) -> None:
        with self.assertRaisesRegex(ModelError, "train one with scripts/train_model.py"):
            newest_model(self.root / "missing")


if __name__ == "__main__":
    unittest.main()
```

Create `tests/app/__init__.py` as an empty file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.app.test_paths_and_models`
Expected: ERROR with `ModuleNotFoundError: No module named 'frog_classifier.app'`.

- [ ] **Step 3: Implement**

`src/frog_classifier/app/__init__.py`:

```python
"""The site staff app's services: deployments, scans, uploads, listening, and export.

Nothing is re-exported here; import modules directly. Only scan_job, which runs in
its own process, imports PyTorch, so the app itself starts quickly and stays small.
"""
```

`src/frog_classifier/app/paths.py`:

```python
"""Where the app keeps models, uploaded recordings, and detection records."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_VARIABLE = "FROG_CLASSIFIER_ROOT"


@dataclass(frozen=True)
class AppPaths:
    root: Path

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def uploads(self) -> Path:
        return self.root / "raw" / "uploads"

    @property
    def detections(self) -> Path:
        return self.root / "results" / "detections"


def app_paths(default_root: Path) -> AppPaths:
    """The data root named by FROG_CLASSIFIER_ROOT, or else default_root."""
    configured = os.environ.get(ROOT_VARIABLE)
    return AppPaths(Path(configured) if configured else default_root)
```

`src/frog_classifier/app/models.py`:

```python
"""The saved model the app scans with, found without loading PyTorch."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from frog_classifier.training.artifact import RUN_METADATA, read_run


# Run folders are named <architecture>-<UTC training time>, such as resnet18-20261014T114432Z.
_TRAINED_AT = re.compile(r"-(\d{8}T\d{6}Z)$")


class ModelError(ValueError):
    pass


@dataclass(frozen=True)
class ModelInfo:
    run_dir: Path
    name: str
    threshold: float


def newest_model(models_root: Path) -> ModelInfo:
    """The most recently trained saved run under models_root.

    Only run.json is read here; the scan process checks every file of the run
    before it uses the model.
    """
    candidates = []
    for run_dir in sorted(models_root.iterdir()) if models_root.is_dir() else ():
        trained_at = _TRAINED_AT.search(run_dir.name)
        if trained_at is None or not (run_dir / RUN_METADATA).is_file():
            continue
        try:
            metadata = read_run(run_dir)
            model = ModelInfo(run_dir, str(metadata["run_name"]), float(metadata["threshold"]["value"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        candidates.append((trained_at.group(1), model))
    if not candidates:
        raise ModelError(f"there is no saved model in {models_root}; train one with scripts/train_model.py")
    return max(candidates, key=lambda candidate: candidate[0])[1]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.app.test_paths_and_models`
Expected: all five tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/app tests/app
git commit -m "feat: find the app's data root and newest model without PyTorch"
```

---

### Task 3: The scan status file

**Files:**
- Create: `src/frog_classifier/app/scans.py`
- Test: `tests/app/test_scans.py`

**Interfaces:**
- Produces: `STATUS_FILE`, `STOP_FILE`, `LOG_FILE`, `HEARTBEAT_SECONDS`, `STALE_AFTER`, `ACTIVE_STATES`; `ScanStatus(state, started_at, updated_at, results_ready, found=0, already_done=0, processed=0, failed=0, device=None, message=None)` with property `done`; `utc_timestamp(moment) -> str`; `read_status(record_dir) -> ScanStatus | None`; `write_status(record_dir, status)`; `current_status(record_dir, now) -> ScanStatus | None`; `stop_requested`, `request_stop`, `clear_stop_request`; `launch_scan_process(run_dir, input_root, record_dir, *, workers=None) -> subprocess.Popen`.

- [ ] **Step 1: Write the failing tests**

Create `tests/app/test_scans.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from frog_classifier.app.scans import (
    STATUS_FILE,
    ScanStatus,
    clear_stop_request,
    current_status,
    read_status,
    request_stop,
    stop_requested,
    write_status,
)
from frog_classifier.files import FileInUseError, replace_file


MOMENT = datetime(2026, 10, 15, tzinfo=timezone.utc)


def status(**changes: object) -> ScanStatus:
    base = ScanStatus(
        state="running",
        started_at="2026-10-15T00:00:00Z",
        updated_at="2026-10-15T00:00:00Z",
        results_ready=False,
        found=10,
        already_done=2,
        processed=3,
    )
    return replace(base, **changes)


class ScanStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.record = Path(directory.name)

    def test_a_written_status_reads_back_unchanged(self) -> None:
        write_status(self.record, status(device="cuda"))

        self.assertEqual(read_status(self.record), status(device="cuda"))
        self.assertEqual(read_status(self.record).done, 5)

    def test_a_missing_or_damaged_status_reads_as_none(self) -> None:
        self.assertIsNone(read_status(self.record))
        (self.record / STATUS_FILE).write_bytes(b"{not json")
        self.assertIsNone(read_status(self.record))

    def test_an_active_status_that_stopped_updating_reads_as_interrupted(self) -> None:
        write_status(self.record, status())

        self.assertEqual(current_status(self.record, MOMENT + timedelta(seconds=29)).state, "running")
        self.assertEqual(current_status(self.record, MOMENT + timedelta(seconds=31)).state, "interrupted")
        write_status(self.record, status(state="finished"))
        self.assertEqual(current_status(self.record, MOMENT + timedelta(hours=1)).state, "finished")

    def test_a_status_being_swapped_in_is_read_again(self) -> None:
        write_status(self.record, status())
        content = (self.record / STATUS_FILE).read_bytes()

        with patch.object(Path, "read_bytes", side_effect=[PermissionError(13, "in use"), content]):
            with patch("frog_classifier.app.scans.time.sleep"):
                self.assertEqual(read_status(self.record), status())

    def test_a_status_file_held_open_is_written_again(self) -> None:
        attempts = []

        def held_open_twice(path, write):
            attempts.append(path)
            if len(attempts) < 3:
                raise FileInUseError("in use")
            replace_file(path, write)

        with patch("frog_classifier.app.scans.replace_file", side_effect=held_open_twice):
            with patch("frog_classifier.app.scans.time.sleep"):
                write_status(self.record, status())

        self.assertEqual(len(attempts), 3)
        self.assertEqual(read_status(self.record), status())

    def test_a_stop_request_is_seen_and_cleared(self) -> None:
        self.assertFalse(stop_requested(self.record))
        request_stop(self.record)
        self.assertTrue(stop_requested(self.record))
        clear_stop_request(self.record)
        self.assertFalse(stop_requested(self.record))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.app.test_scans`
Expected: ERROR with `ModuleNotFoundError: No module named 'frog_classifier.app.scans'`.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/app/scans.py`:

```python
"""Scan processes and the status file through which they report.

A scan runs in its own process, so it survives a page refresh and keeps PyTorch
out of the app. It reports through scan_status.json in its record folder, and a
stop is requested by creating stop_requested there.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from frog_classifier.files import FileInUseError, replace_file


STATUS_FILE = "scan_status.json"
STOP_FILE = "stop_requested"
LOG_FILE = "scan.log"
HEARTBEAT_SECONDS = 2.0
# A starting or running scan rewrites its status every two seconds, so one this
# old belongs to a process that ended without saying so.
STALE_AFTER = timedelta(seconds=30)
ACTIVE_STATES = frozenset({"starting", "running"})
_ATTEMPTS = 50
_RETRY_SECONDS = 0.1

# Kept so a scan process still running is never reported as abandoned.
_launched: list[subprocess.Popen] = []


@dataclass(frozen=True)
class ScanStatus:
    """A scan's progress; results_ready says whether the record holds complete results."""

    state: str
    started_at: str
    updated_at: str
    results_ready: bool
    found: int = 0
    already_done: int = 0
    processed: int = 0
    failed: int = 0
    device: str | None = None
    message: str | None = None

    @property
    def done(self) -> int:
        return self.already_done + self.processed


def utc_timestamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_status(record_dir: Path) -> ScanStatus | None:
    """The last status written in record_dir, or None when there is none."""
    path = record_dir / STATUS_FILE
    for attempt in range(_ATTEMPTS):
        try:
            content = path.read_bytes()
            break
        except FileNotFoundError:
            return None
        except PermissionError:
            # The scan process is swapping in a new status at this moment.
            if attempt == _ATTEMPTS - 1:
                raise
            time.sleep(_RETRY_SECONDS)
    try:
        return ScanStatus(**json.loads(content))
    except (ValueError, TypeError):
        return None


def write_status(record_dir: Path, status: ScanStatus) -> None:
    """Swap in a new status, retrying while a reader holds the old one open."""
    content = json.dumps(asdict(status), indent=2).encode("utf-8")
    for attempt in range(_ATTEMPTS):
        try:
            replace_file(record_dir / STATUS_FILE, lambda staging: staging.write_bytes(content))
            return
        except FileInUseError:
            if attempt == _ATTEMPTS - 1:
                raise
            time.sleep(_RETRY_SECONDS)


def current_status(record_dir: Path, now: datetime) -> ScanStatus | None:
    """The status as it stands at now; an active scan that stopped reporting is interrupted."""
    status = read_status(record_dir)
    if status is not None and status.state in ACTIVE_STATES:
        if now - datetime.fromisoformat(status.updated_at) > STALE_AFTER:
            return replace(status, state="interrupted")
    return status


def stop_requested(record_dir: Path) -> bool:
    return (record_dir / STOP_FILE).exists()


def request_stop(record_dir: Path) -> None:
    (record_dir / STOP_FILE).touch()


def clear_stop_request(record_dir: Path) -> None:
    (record_dir / STOP_FILE).unlink(missing_ok=True)


def launch_scan_process(
    run_dir: Path,
    input_root: Path,
    record_dir: Path,
    *,
    workers: int | None = None,
) -> subprocess.Popen:
    """Start the scan process; its output goes to scan.log in record_dir."""
    command = [sys.executable, "-m", "frog_classifier.app.scan_job", str(run_dir), str(input_root), str(record_dir)]
    if workers is not None:
        command += ["--workers", str(workers)]
    with (record_dir / LOG_FILE).open("ab") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    _launched[:] = [*(running for running in _launched if running.poll() is None), process]
    return process
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.app.test_scans`
Expected: all six tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/app/scans.py tests/app/test_scans.py
git commit -m "feat: report scan progress through a status file"
```

---

### Task 4: Deployments

**Files:**
- Create: `src/frog_classifier/app/deployments.py`
- Modify: `src/frog_classifier/inference/recordings.py`
- Test: `tests/app/test_deployments.py`

**Interfaces:**
- Consumes: Task 3 status functions, `ModelInfo`, `AppPaths`.
- Produces: `start_time_from_name(stem) -> datetime | None` in `inference.recordings`; `DEPLOYMENT_FILE`, `NAME_LIMIT`, `DeploymentError(ValueError)`, `Deployment(name, record_dir, input_root, model_run, created_at, status=None)` with properties `scanning` and `results_ready`; `RecordingsPreview(count, first_day, last_day, undated)`; `clean_name`, `deployment_dir`, `list_deployments(detections_root, *, now)`, `live_scan(deployments)`, `preview_recordings(folder)`, `start_scan(paths, model, folder, name, *, now, workers=None) -> Path`.

- [ ] **Step 1: Write the failing tests**

Create `tests/app/test_deployments.py`:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from frog_classifier.app.deployments import (
    DEPLOYMENT_FILE,
    DeploymentError,
    RecordingsPreview,
    clean_name,
    deployment_dir,
    list_deployments,
    preview_recordings,
    start_scan,
)
from frog_classifier.app.models import ModelInfo
from frog_classifier.app.paths import AppPaths
from frog_classifier.app.scans import ScanStatus, request_stop, stop_requested, write_status
from tests.analytics.helpers import build_record, recording
from tests.inference.helpers import SCAN_IDENTITY


MODEL = ModelInfo(Path("models/resnet18-20261015T000000Z"), "resnet18-20261015T000000Z", 0.5)
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


def finished(results_ready: bool = True) -> ScanStatus:
    return ScanStatus("finished", "2026-10-15T12:00:00Z", "2026-10-15T12:01:00Z", results_ready, found=3, processed=3)


class NameTests(unittest.TestCase):
    def test_names_keep_only_characters_safe_in_folder_names(self) -> None:
        for text, expected in (
            ("IWS076 card 3", "IWS076 card 3"),
            ("site/a: north", "site-a- north"),
            ("  card 1.  ", "card 1"),
            ("Pond (east)_2", "Pond (east)_2"),
        ):
            with self.subTest(text=text):
                self.assertEqual(clean_name(text), expected)

    def test_unusable_names_are_refused(self) -> None:
        for text in ("", " . ", "CON", "lpt1.txt", "x" * 81):
            with self.subTest(text=text), self.assertRaises(DeploymentError):
                clean_name(text)

    def test_a_deployment_folder_ends_with_the_model_run(self) -> None:
        self.assertEqual(deployment_dir(Path("results"), "card 3", "resnet18-x"), Path("results") / "card 3-resnet18-x")


class DeploymentTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.paths = AppPaths(self.root)
        self.card = self.root / "card"
        for name in ("20220123_010000.WAV", "b/20220405_020000.wav", "notes.txt", "c/unnamed.mp3"):
            (self.card / name).parent.mkdir(parents=True, exist_ok=True)
            (self.card / name).write_bytes(b"")

    def _start(self, name: str = "card", folder: Path | None = None, now: datetime = NOW):
        with patch("frog_classifier.app.deployments.launch_scan_process") as launch:
            record_dir = start_scan(self.paths, MODEL, folder or self.card, name, now=now)
        return record_dir, launch

    def test_starting_a_scan_records_the_deployment_and_launches_the_scan(self) -> None:
        record_dir, launch = self._start()

        self.assertEqual(record_dir, self.paths.detections / "card-resnet18-20261015T000000Z")
        launch.assert_called_once_with(MODEL.run_dir, self.card.resolve(), record_dir, workers=None)
        [deployment] = list_deployments(self.paths.detections, now=NOW)
        self.assertEqual(
            (deployment.name, deployment.input_root, deployment.model_run, deployment.created_at),
            ("card", self.card.resolve().as_posix(), MODEL.name, "2026-10-15T12:00:00Z"),
        )
        self.assertEqual(deployment.status.state, "starting")
        self.assertTrue(deployment.scanning)
        self.assertFalse(deployment.results_ready)

    def test_continuing_keeps_the_deployment_and_its_complete_results(self) -> None:
        record_dir, _ = self._start()
        write_status(record_dir, finished())
        request_stop(record_dir)
        later = NOW + timedelta(days=1)

        self._start(now=later)

        [deployment] = list_deployments(self.paths.detections, now=later)
        self.assertEqual(deployment.created_at, "2026-10-15T12:00:00Z")
        self.assertEqual((deployment.status.state, deployment.status.results_ready), ("starting", True))
        self.assertFalse(stop_requested(record_dir))

    def test_a_name_used_for_another_folder_is_refused(self) -> None:
        record_dir, _ = self._start()
        write_status(record_dir, finished())
        other = self.root / "other"
        other.mkdir()

        with self.assertRaisesRegex(DeploymentError, "already used for the recordings in .*card; choose another name"):
            self._start(folder=other)

    def test_only_one_scan_runs_at_a_time(self) -> None:
        self._start()
        other = self.root / "other"
        other.mkdir()

        with self.assertRaisesRegex(DeploymentError, "the scan of card is still running"):
            self._start("other", other)

    def test_a_missing_folder_is_refused(self) -> None:
        with self.assertRaisesRegex(DeploymentError, "there is no folder"):
            self._start("missing", self.root / "missing")

    def test_records_made_by_the_command_are_listed_by_their_folder_name(self) -> None:
        build_record(self.paths.detections / f"external-{SCAN_IDENTITY.model_run}", [
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.9,)),
        ])
        self._start()

        deployments = list_deployments(self.paths.detections, now=NOW)

        self.assertEqual([deployment.name for deployment in deployments], ["card", "external"])
        command = deployments[1]
        self.assertEqual((command.status, command.input_root), (None, SCAN_IDENTITY.input_root))
        self.assertTrue(command.results_ready)

    def test_a_scan_that_stopped_reporting_is_interrupted_and_incomplete(self) -> None:
        self._start()

        [deployment] = list_deployments(self.paths.detections, now=NOW + timedelta(minutes=5))

        self.assertEqual(deployment.status.state, "interrupted")
        self.assertFalse(deployment.scanning)
        self.assertFalse(deployment.results_ready)

    def test_a_damaged_deployment_file_is_skipped(self) -> None:
        record_dir, _ = self._start()
        (record_dir / DEPLOYMENT_FILE).write_bytes(b"{not json")

        self.assertEqual(list_deployments(self.paths.detections, now=NOW), [])

    def test_a_folder_preview_counts_recordings_and_their_dates(self) -> None:
        self.assertEqual(
            preview_recordings(self.card),
            RecordingsPreview(count=3, first_day=date(2022, 1, 23), last_day=date(2022, 4, 5), undated=1),
        )

    def test_previewing_a_missing_folder_is_refused(self) -> None:
        with self.assertRaisesRegex(DeploymentError, "there is no folder"):
            preview_recordings(self.root / "missing")

    def test_the_deployment_file_is_plain_json(self) -> None:
        record_dir, _ = self._start()

        self.assertEqual(json.loads((record_dir / DEPLOYMENT_FILE).read_bytes())["name"], "card")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.app.test_deployments`
Expected: ERROR with `ModuleNotFoundError: No module named 'frog_classifier.app.deployments'`.

- [ ] **Step 3: Implement**

Add to `src/frog_classifier/inference/recordings.py` after `recording_time`:

```python
def start_time_from_name(stem: str) -> datetime | None:
    """The start time given by a YYYYMMDD_HHMMSS file name, or None."""
    return _file_name_time(stem)[0]
```

Create `src/frog_classifier/app/deployments.py`:

```python
"""Deployments: named scans of one folder of recordings, each with its own detection record."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path, PurePosixPath

from frog_classifier.files import replace_file
from frog_classifier.inference.recordings import find_recordings, start_time_from_name
from frog_classifier.inference.store import DATABASE_FILE, DetectionStore, StoreError

from .models import ModelInfo
from .paths import AppPaths
from .scans import (
    ACTIVE_STATES,
    ScanStatus,
    clear_stop_request,
    current_status,
    launch_scan_process,
    read_status,
    utc_timestamp,
    write_status,
)


DEPLOYMENT_FILE = "deployment.json"
NAME_LIMIT = 80
_UNSAFE_CHARACTERS = re.compile(r"[^A-Za-z0-9 _.()-]+")
_RESERVED_NAMES = frozenset({
    "CON", "PRN", "AUX", "NUL", *(f"COM{number}" for number in range(1, 10)), *(f"LPT{number}" for number in range(1, 10)),
})


class DeploymentError(ValueError):
    pass


@dataclass(frozen=True)
class Deployment:
    name: str
    record_dir: Path
    input_root: str
    model_run: str
    created_at: str
    status: ScanStatus | None = None

    @property
    def scanning(self) -> bool:
        return self.status is not None and self.status.state in ACTIVE_STATES

    @property
    def results_ready(self) -> bool:
        """Whether the record holds complete results; a record made by the detection command has no status."""
        if self.status is None:
            return True
        return not self.scanning and self.status.results_ready


@dataclass(frozen=True)
class RecordingsPreview:
    count: int
    first_day: date | None
    last_day: date | None
    undated: int


def clean_name(text: str) -> str:
    """A deployment name that is safe as part of a folder name on every system."""
    # Windows drops trailing dots and spaces from folder names.
    name = _UNSAFE_CHARACTERS.sub("-", text).strip(" .")
    if not name:
        raise DeploymentError("give these recordings a name, such as the site and card number")
    if len(name) > NAME_LIMIT:
        raise DeploymentError(f"keep the name to {NAME_LIMIT} characters or fewer")
    if name.split(".")[0].upper() in _RESERVED_NAMES:
        raise DeploymentError(f"{name!r} is reserved by Windows; choose another name")
    return name


def deployment_dir(detections_root: Path, name: str, model_name: str) -> Path:
    return detections_root / f"{clean_name(name)}-{model_name}"


def list_deployments(detections_root: Path, *, now: datetime) -> list[Deployment]:
    """Every deployment with a record under detections_root, newest first."""
    if not detections_root.is_dir():
        return []
    deployments = []
    for record_dir in detections_root.iterdir():
        deployment = _read_deployment(record_dir) if record_dir.is_dir() else None
        if deployment is not None:
            deployments.append(replace(deployment, status=current_status(record_dir, now)))
    return sorted(deployments, key=lambda deployment: deployment.created_at, reverse=True)


def live_scan(deployments: list[Deployment]) -> Deployment | None:
    return next((deployment for deployment in deployments if deployment.scanning), None)


def preview_recordings(folder: Path) -> RecordingsPreview:
    """How many recordings a folder holds, and the dates their file names give."""
    if not folder.is_dir():
        raise DeploymentError(f"there is no folder {folder}")
    paths = find_recordings(folder)
    days = [
        start.date()
        for path in paths
        if (start := start_time_from_name(PurePosixPath(path).stem)) is not None
    ]
    return RecordingsPreview(len(paths), min(days, default=None), max(days, default=None), len(paths) - len(days))


def start_scan(
    paths: AppPaths,
    model: ModelInfo,
    folder: Path,
    name: str,
    *,
    now: datetime,
    workers: int | None = None,
) -> Path:
    """Start scanning folder as the deployment called name, or continue that deployment.

    Returns the deployment's record folder; the scan runs in its own process.
    """
    name = clean_name(name)
    folder = folder.resolve()
    if not folder.is_dir():
        raise DeploymentError(f"there is no folder {folder}")
    deployments = list_deployments(paths.detections, now=now)
    running = live_scan(deployments)
    if running is not None:
        raise DeploymentError(f"the scan of {running.name} is still running; wait for it to finish or stop it first")
    record_dir = deployment_dir(paths.detections, name, model.name)
    existing = next((deployment for deployment in deployments if deployment.record_dir == record_dir), None)
    if existing is not None and Path(existing.input_root) != folder:
        raise DeploymentError(
            f"the name {name!r} is already used for the recordings in {existing.input_root}; choose another name"
        )
    record_dir.mkdir(parents=True, exist_ok=True)
    if existing is None:
        _write_deployment(Deployment(name, record_dir, folder.as_posix(), model.name, utc_timestamp(now)))
    previous = read_status(record_dir)
    started = utc_timestamp(now)
    clear_stop_request(record_dir)
    write_status(record_dir, ScanStatus(
        state="starting",
        started_at=started,
        updated_at=started,
        results_ready=previous.results_ready if previous is not None else (record_dir / DATABASE_FILE).is_file(),
    ))
    launch_scan_process(model.run_dir, folder, record_dir, workers=workers)
    return record_dir


def _read_deployment(record_dir: Path) -> Deployment | None:
    """The deployment the app recorded, or else one made by the detection command."""
    try:
        data = json.loads((record_dir / DEPLOYMENT_FILE).read_bytes())
        return Deployment(record_dir=record_dir, **data)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError):
        return None
    if not (record_dir / DATABASE_FILE).is_file():
        return None
    try:
        with DetectionStore.open_read_only(record_dir / DATABASE_FILE) as store:
            identity, created_at = store.identity, store.created_at
    except StoreError:
        return None
    name = record_dir.name.removesuffix(f"-{identity.model_run}")
    return Deployment(name, record_dir, identity.input_root, identity.model_run, created_at)


def _write_deployment(deployment: Deployment) -> None:
    content = json.dumps({
        "name": deployment.name,
        "input_root": deployment.input_root,
        "model_run": deployment.model_run,
        "created_at": deployment.created_at,
    }, indent=2).encode("utf-8")
    replace_file(deployment.record_dir / DEPLOYMENT_FILE, lambda staging: staging.write_bytes(content))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.app.test_deployments tests.inference.test_recordings`
Expected: every test passes.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/app/deployments.py src/frog_classifier/inference/recordings.py tests/app/test_deployments.py
git commit -m "feat: name, list, and start deployments"
```

---

### Task 5: The scan process

**Files:**
- Create: `src/frog_classifier/app/scan_job.py`
- Test: `tests/app/test_scan_job.py`

**Interfaces:**
- Consumes: Task 3 status functions and `launch_scan_process`; `detect_calls`, `default_workers`, `select_device`.
- Produces: `python -m frog_classifier.app.scan_job RUN INPUT RECORD [--workers N]`, exiting 0 when finished, 130 when stopped, and 1 when failed; `main(argv) -> int`.

- [ ] **Step 1: Write the failing tests**

Create `tests/app/test_scan_job.py`:

```python
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from frog_classifier.app.scans import ScanStatus, launch_scan_process, read_status, request_stop, write_status
from frog_classifier.inference.store import DATABASE_FILE, DetectionStore
from tests.inference.helpers import AUDIOMOTH_COMMENT, build_saved_run, write_recording
from tests.preprocessing.helpers import noise


TIMEOUT_S = 300


class ScanJobTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.run_dir = build_saved_run(self.root / "models")
        self.card = self.root / "card"
        for seed, name in enumerate(("20220123_010000.WAV", "20220123_020000.WAV", "20220124_010000.WAV")):
            write_recording(self.card / name, noise(6, sample_rate_hz=48000, seed=seed), comment=AUDIOMOTH_COMMENT)
        self.record = self.root / "results" / "detections" / "card-run"
        self.record.mkdir(parents=True)

    def _run(self, input_root: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "frog_classifier.app.scan_job", str(self.run_dir), str(input_root), str(self.record),
             "--workers", "1"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )

    def _stored(self) -> int:
        if not (self.record / DATABASE_FILE).is_file():
            return 0
        with DetectionStore.open_read_only(self.record / DATABASE_FILE) as store:
            return len(store.recording_rows())

    def test_a_scan_stores_every_recording_and_reports_finished(self) -> None:
        result = self._run(self.card)

        self.assertEqual(result.returncode, 0, result.stderr)
        status = read_status(self.record)
        self.assertEqual(
            (status.state, status.found, status.processed, status.failed, status.results_ready),
            ("finished", 3, 3, 0, True),
        )
        self.assertIn(status.device, ("cpu", "cuda"))
        self.assertEqual(self._stored(), 3)

    def test_a_stop_request_keeps_only_whole_recordings(self) -> None:
        process = launch_scan_process(self.run_dir, self.card, self.record, workers=1)
        # Requested before the process has loaded PyTorch, so it never reaches the last recording.
        request_stop(self.record)

        self.assertEqual(process.wait(timeout=TIMEOUT_S), 130)
        status = read_status(self.record)
        self.assertEqual((status.state, status.results_ready), ("stopped", False))
        self.assertLess(self._stored(), 3)

    def test_a_scan_that_cannot_start_says_why_and_keeps_earlier_results(self) -> None:
        write_status(self.record, ScanStatus("starting", "2026-10-15T00:00:00Z", "2026-10-15T00:00:00Z", True))

        result = self._run(self.root / "missing")

        self.assertEqual(result.returncode, 1)
        status = read_status(self.record)
        self.assertEqual((status.state, status.results_ready), ("failed", True))
        self.assertIn("input folder not found", status.message)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.app.test_scan_job`
Expected: FAIL, because `python -m frog_classifier.app.scan_job` exits with 1 and "No module named frog_classifier.app.scan_job", and no status is written.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/app/scan_job.py`:

```python
"""The scan process: detect calls in one folder and keep its status file current.

The app starts it as python -m frog_classifier.app.scan_job RUN INPUT RECORD.
PyTorch loads only after the status thread is running, so the app sees the scan
within seconds of starting it.
"""

from __future__ import annotations

import _thread
import argparse
import threading
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from frog_classifier.files import FileInUseError
from frog_classifier.inference.store import DATABASE_FILE

from .scans import HEARTBEAT_SECONDS, ScanStatus, read_status, stop_requested, utc_timestamp, write_status


STOPPED = "Stopped on request; finished recordings are kept."
_EXIT_CODES = {"finished": 0, "stopped": 130}


class ScanStopped(Exception):
    """A stop was requested between two recordings."""


class StatusFile:
    """The scan's status, rewritten on every change; safe to update from two threads."""

    def __init__(self, record_dir: Path, status: ScanStatus) -> None:
        self.record_dir = record_dir
        self._status = status
        self._lock = threading.Lock()
        write_status(record_dir, status)

    @property
    def status(self) -> ScanStatus:
        with self._lock:
            return self._status

    def update(self, **changes: object) -> None:
        with self._lock:
            self._status = replace(self._status, **changes, updated_at=utc_timestamp(datetime.now(timezone.utc)))
            write_status(self.record_dir, self._status)

    def beat(self) -> None:
        """Show that the scan is alive; a reader holding the file only delays this to the next beat."""
        try:
            self.update()
        except FileInUseError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("run", type=Path, help="Saved model run folder.")
    parser.add_argument("input", type=Path, help="Folder of recordings.")
    parser.add_argument("record", type=Path, help="The deployment's detection record folder.")
    parser.add_argument("--workers", type=int, default=None, help="Rendering processes. Default: as the detection command.")
    args = parser.parse_args(argv)

    previous = read_status(args.record)
    started = utc_timestamp(datetime.now(timezone.utc))
    status = StatusFile(args.record, ScanStatus(
        state="starting",
        started_at=started,
        updated_at=started,
        results_ready=previous.results_ready if previous is not None else (args.record / DATABASE_FILE).is_file(),
    ))
    finished = threading.Event()
    heartbeat = threading.Thread(target=_keep_alive, args=(status, finished), daemon=True)
    heartbeat.start()
    try:
        try:
            outcome = _scan(args, status)
        finally:
            finished.set()
            heartbeat.join()
    except KeyboardInterrupt:
        outcome = {"state": "stopped", "message": STOPPED}
    except Exception as error:
        traceback.print_exc()
        outcome = {"state": "failed", "message": f"{type(error).__name__}: {error}"}
    status.update(**outcome)
    return _EXIT_CODES.get(str(outcome["state"]), 1)


def _scan(args: argparse.Namespace, status: StatusFile) -> dict[str, object]:
    # Imported here: the status thread is already running while PyTorch loads.
    from frog_classifier.inference.detect import DetectionError, default_workers, detect_calls
    from frog_classifier.inference.store import StoreError
    from frog_classifier.training.fit import select_device
    from frog_classifier.training.saved import SavedRunError

    device = select_device()
    status.update(device=device.type)

    def started(pending: int, done: int) -> None:
        # The record is complete only while nothing new is waiting to be scanned.
        status.update(
            state="running",
            found=pending + done,
            already_done=done,
            processed=0,
            failed=0,
            results_ready=status.status.results_ready and pending == 0,
        )

    def stored(outcome) -> None:
        current = status.status
        status.update(processed=current.processed + 1, failed=current.failed + int(outcome.failed))
        if stop_requested(status.record_dir):
            raise ScanStopped

    try:
        summary = detect_calls(
            args.run,
            args.input,
            args.record,
            device=device,
            workers=args.workers or default_workers(),
            on_start=started,
            on_recording=stored,
        )
    except ScanStopped:
        return {"state": "stopped", "message": STOPPED}
    except (DetectionError, SavedRunError, StoreError) as error:
        # Refused before anything was scanned, so the record is as complete as before.
        return {"state": "failed", "message": str(error)}
    return {"state": "finished", "results_ready": True, "failed": summary.failed, "message": summary.csv_problem}


def _keep_alive(status: StatusFile, finished: threading.Event) -> None:
    interrupted = False
    while not finished.wait(HEARTBEAT_SECONDS):
        status.beat()
        if not interrupted and stop_requested(status.record_dir):
            # Reaches a long render too; the main thread treats it like Ctrl+C.
            interrupted = True
            _thread.interrupt_main()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.app.test_scan_job tests.app.test_scans`
Expected: every test passes.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/app/scan_job.py tests/app/test_scan_job.py
git commit -m "feat: run each scan in its own process with a live status"
```

---

### Task 6: Uploads

**Files:**
- Create: `src/frog_classifier/app/uploads.py`
- Test: `tests/app/test_uploads.py`

**Interfaces:**
- Produces: `UploadError(ValueError)`, `SavedUploads(saved, skipped)`, `save_uploads(files: Sequence[tuple[str, bytes]], folder) -> SavedUploads`.

- [ ] **Step 1: Write the failing tests**

Create `tests/app/test_uploads.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from frog_classifier.app.uploads import SavedUploads, UploadError, save_uploads


class SaveUploadsTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name) / "uploads" / "card"

    def test_recordings_are_saved_under_their_own_names(self) -> None:
        result = save_uploads([("C:\\fakepath\\20220123_010000.WAV", b"one"), ("site/b.mp3", b"two")], self.folder)

        self.assertEqual(result, SavedUploads(saved=("20220123_010000.WAV", "b.mp3"), skipped=()))
        self.assertEqual((self.folder / "20220123_010000.WAV").read_bytes(), b"one")
        self.assertEqual((self.folder / "b.mp3").read_bytes(), b"two")

    def test_an_identical_recording_is_skipped(self) -> None:
        save_uploads([("a.wav", b"one")], self.folder)

        result = save_uploads([("a.wav", b"one"), ("b.wav", b"two")], self.folder)

        self.assertEqual(result, SavedUploads(saved=("b.wav",), skipped=("a.wav",)))

    def test_a_different_recording_under_a_saved_name_is_refused_and_nothing_is_saved(self) -> None:
        save_uploads([("a.wav", b"one")], self.folder)

        with self.assertRaisesRegex(UploadError, "already holds different recordings named a.wav"):
            save_uploads([("b.wav", b"two"), ("a.wav", b"changed")], self.folder)

        self.assertEqual((self.folder / "a.wav").read_bytes(), b"one")
        self.assertFalse((self.folder / "b.wav").exists())

    def test_files_that_are_not_recordings_are_refused(self) -> None:
        with self.assertRaisesRegex(UploadError, "only .wav and .mp3 recordings can be scanned, not notes.txt"):
            save_uploads([("notes.txt", b"x")], self.folder)

    def test_one_name_uploaded_twice_is_refused(self) -> None:
        with self.assertRaisesRegex(UploadError, "a.wav is in this upload more than once"):
            save_uploads([("x/a.wav", b"one"), ("y/A.WAV", b"two")], self.folder)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.app.test_uploads`
Expected: ERROR with `ModuleNotFoundError: No module named 'frog_classifier.app.uploads'`.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/app/uploads.py`:

```python
"""Save uploaded recordings into a deployment's upload folder without overwriting any."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePath, PureWindowsPath
from typing import Sequence

from frog_classifier.files import replace_file
from frog_classifier.inference.recordings import AUDIO_SUFFIXES


class UploadError(ValueError):
    pass


@dataclass(frozen=True)
class SavedUploads:
    saved: tuple[str, ...]
    skipped: tuple[str, ...]


def save_uploads(files: Sequence[tuple[str, bytes]], folder: Path) -> SavedUploads:
    """Save each (file name, content) in folder.

    A file already there with the same content is skipped. One with different
    content is refused, and then nothing is saved, so a recording is never
    overwritten.
    """
    # Browsers may send folder parts, with either separator; only the file name is kept.
    named = [(PureWindowsPath(name).name, content) for name, content in files]
    unsupported = [name for name, _ in named if PurePath(name).suffix.lower() not in AUDIO_SUFFIXES]
    if unsupported:
        raise UploadError(f"only .wav and .mp3 recordings can be scanned, not {', '.join(unsupported)}")
    repeated = [name for name, count in Counter(name.lower() for name, _ in named).items() if count > 1]
    if repeated:
        raise UploadError(f"{', '.join(repeated)} is in this upload more than once")
    conflicts = [
        name for name, content in named
        if (folder / name).is_file() and (folder / name).read_bytes() != content
    ]
    if conflicts:
        raise UploadError(
            f"{folder} already holds different recordings named {', '.join(conflicts)}; "
            "rename them or use another name for these recordings"
        )
    folder.mkdir(parents=True, exist_ok=True)
    saved, skipped = [], []
    for name, content in named:
        target = folder / name
        if target.is_file():
            skipped.append(name)
            continue
        replace_file(target, lambda staging, content=content: staging.write_bytes(content))
        saved.append(name)
    return SavedUploads(tuple(saved), tuple(skipped))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.app.test_uploads`
Expected: all five tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/app/uploads.py tests/app/test_uploads.py
git commit -m "feat: save uploaded recordings without overwriting any"
```

---

### Task 7: Listening to a window

**Files:**
- Create: `src/frog_classifier/app/listening.py`
- Test: `tests/app/test_listening.py`

**Interfaces:**
- Produces: `ListeningError(ValueError)`, `WindowView(wav: bytes, image: np.ndarray)`, `model_preprocessing(models_root, model_run) -> PreprocessingConfig`, `load_window(recording, start_s, config) -> WindowView`.

- [ ] **Step 1: Write the failing tests**

Create `tests/app/test_listening.py`:

```python
from __future__ import annotations

import io
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from frog_classifier.app.listening import ListeningError, load_window, model_preprocessing
from frog_classifier.data.config import load_preprocessing_config
from frog_classifier.preprocessing import load_waveform, render_waveform_images
from frog_classifier.preprocessing.display import colorize
from frog_classifier.preprocessing.spectrogram import encode_png
from frog_classifier.training.artifact import PREPROCESSING_FILE
from tests.data.helpers import VALID_CONFIG
from tests.inference.helpers import write_recording
from tests.preprocessing.helpers import noise


class ListeningTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        run_dir = self.root / "models" / "resnet18-20261015T000000Z"
        run_dir.mkdir(parents=True)
        (run_dir / PREPROCESSING_FILE).write_bytes(VALID_CONFIG)
        self.config = load_preprocessing_config(run_dir / PREPROCESSING_FILE)
        self.recording = write_recording(self.root / "card" / "20220123_010000.WAV", noise(12, sample_rate_hz=48000))

    def test_a_window_plays_its_five_seconds_beside_the_image_the_model_saw(self) -> None:
        view = load_window(self.recording, 5, self.config)

        rate = self.config.audio.sample_rate_hz
        with wave.open(io.BytesIO(view.wav)) as sound:
            self.assertEqual((sound.getnchannels(), sound.getframerate()), (1, rate))
            frames = np.frombuffer(sound.readframes(sound.getnframes()), dtype="<i2")
        self.assertEqual(len(frames), 5 * rate)
        self.assertEqual(int(np.max(np.abs(frames))), 32767)
        waveform = load_waveform(self.recording, self.config)
        expected = next(image for image in render_waveform_images(waveform, self.config) if image.start_s == 5)
        np.testing.assert_array_equal(view.image, colorize(encode_png(expected.pixels)))

    def test_a_recording_that_is_no_longer_there_says_where_it_was_expected(self) -> None:
        with self.assertRaisesRegex(ListeningError, "missing.WAV is not there; connect the card or drive"):
            load_window(self.root / "card" / "missing.WAV", 0, self.config)

    def test_a_window_past_the_end_is_refused(self) -> None:
        with self.assertRaisesRegex(ListeningError, "no five-second window starting at 10 seconds"):
            load_window(self.recording, 10, self.config)

    def test_the_preprocessing_comes_from_the_model_run(self) -> None:
        self.assertEqual(model_preprocessing(self.root / "models", "resnet18-20261015T000000Z"), self.config)
        with self.assertRaisesRegex(ListeningError, "the model resnet18-gone is no longer in"):
            model_preprocessing(self.root / "models", "resnet18-gone")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.app.test_listening`
Expected: ERROR with `ModuleNotFoundError: No module named 'frog_classifier.app.listening'`.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/app/listening.py`:

```python
"""Hear and see one five-second window, as the model saw it."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from frog_classifier.data.config import PreprocessingConfig, load_preprocessing_config
from frog_classifier.preprocessing import load_waveform, render_waveform_images
from frog_classifier.preprocessing.display import colorize
from frog_classifier.preprocessing.spectrogram import encode_png
from frog_classifier.training.artifact import PREPROCESSING_FILE


class ListeningError(ValueError):
    pass


@dataclass(frozen=True)
class WindowView:
    wav: bytes
    image: np.ndarray


def model_preprocessing(models_root: Path, model_run: str) -> PreprocessingConfig:
    """The preprocessing contract saved with a model run."""
    path = models_root / model_run / PREPROCESSING_FILE
    if not path.is_file():
        raise ListeningError(f"the model {model_run} is no longer in {models_root}, so its windows cannot be shown")
    return load_preprocessing_config(path)


def load_window(recording: Path, start_s: int, config: PreprocessingConfig) -> WindowView:
    """The window's sound, raised to full loudness, and the spectrogram the model saw."""
    if not recording.is_file():
        raise ListeningError(f"{recording} is not there; connect the card or drive that holds it to listen")
    waveform = load_waveform(recording, config)
    image = next((chunk for chunk in render_waveform_images(waveform, config) if chunk.start_s == start_s), None)
    if image is None:
        raise ListeningError(f"{recording} has no five-second window starting at {start_s} seconds")
    rate = config.audio.sample_rate_hz
    samples = waveform[start_s * rate:(start_s + config.audio.chunk_seconds) * rate]
    return WindowView(_wav_bytes(samples, rate), colorize(encode_png(image.pixels)))


def _wav_bytes(samples: np.ndarray, sample_rate_hz: int) -> bytes:
    """16-bit mono WAV, scaled so the loudest sample is at full scale and faint calls can be heard."""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    scaled = samples / peak if peak > 0 else samples
    pcm = (np.clip(scaled, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as sound:
        sound.setnchannels(1)
        sound.setsampwidth(2)
        sound.setframerate(sample_rate_hz)
        sound.writeframes(pcm)
    return buffer.getvalue()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.app.test_listening`
Expected: all four tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/app/listening.py tests/app/test_listening.py
git commit -m "feat: play a window and show the spectrogram the model saw"
```

---

### Task 8: The results download

**Files:**
- Create: `src/frog_classifier/app/export.py`
- Test: `tests/app/test_export.py`

**Interfaces:**
- Consumes: `analyze_record`, `OUTPUT_FILES`, `DETECTIONS_CSV`, `RECORDINGS_CSV`.
- Produces: `results_zip(record_dir, *, threshold=None, utc_offset=None) -> bytes`.

- [ ] **Step 1: Write the failing tests**

Create `tests/app/test_export.py`:

```python
from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from frog_classifier.analytics.analyze import OUTPUT_FILES
from frog_classifier.app.export import results_zip
from frog_classifier.inference.export import DETECTIONS_CSV, RECORDINGS_CSV, export_csv
from frog_classifier.inference.store import DATABASE_FILE, DetectionStore
from tests.analytics.helpers import build_record, recording


class ResultsZipTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.record = build_record(Path(directory.name) / "card-run", [
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.95, 0.6, 0.1)),
        ])
        with DetectionStore.open_read_only(self.record / DATABASE_FILE) as store:
            export_csv(store, self.record)

    def test_the_zip_holds_the_analysis_and_both_spreadsheets_in_one_folder(self) -> None:
        with zipfile.ZipFile(io.BytesIO(results_zip(self.record))) as archive:
            names = set(archive.namelist())

        self.assertEqual(names, {f"card-run/{name}" for name in (*OUTPUT_FILES, DETECTIONS_CSV, RECORDINGS_CSV)})

    def test_the_analysis_uses_the_chosen_threshold_and_offset(self) -> None:
        with zipfile.ZipFile(io.BytesIO(results_zip(self.record, threshold=0.9, utc_offset="+10:00"))) as archive:
            report = archive.read("card-run/report.md").decode("utf-8")

        self.assertIn("(1 of 3 windows of 5 seconds) at threshold 0.900", report)
        self.assertIn("| Time basis | UTC+10:00 |", report)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.app.test_export`
Expected: ERROR with `ModuleNotFoundError: No module named 'frog_classifier.app.export'`.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/app/export.py`:

```python
"""The results download: the analysis and the detection spreadsheets in one zip file."""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

from frog_classifier.analytics.analyze import analyze_record
from frog_classifier.inference.export import DETECTIONS_CSV, RECORDINGS_CSV


def results_zip(record_dir: Path, *, threshold: float | None = None, utc_offset: str | None = None) -> bytes:
    """A zip holding one folder named after the record: the analysis at this
    threshold and offset, and the detection spreadsheets."""
    folder = record_dir.resolve().name
    buffer = io.BytesIO()
    with tempfile.TemporaryDirectory() as temporary:
        analysis_dir = Path(temporary) / "analysis"
        analyze_record(record_dir, analysis_dir, threshold=threshold, utc_offset=utc_offset)
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(analysis_dir.iterdir()):
                archive.write(path, f"{folder}/{path.name}")
            for name in (DETECTIONS_CSV, RECORDINGS_CSV):
                if (record_dir / name).is_file():
                    archive.write(record_dir / name, f"{folder}/{name}")
    return buffer.getvalue()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.app.test_export`
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/app/export.py tests/app/test_export.py
git commit -m "feat: download the results as one zip file"
```

---

### Task 9: The interface

**Files:**
- Create: `src/frog_classifier/app/dialog.py`, `classifier_app/app.py`
- Delete: `classifier_app/.gitkeep`
- Modify: `tests/test_repository_reproducibility.py`, `docs/workflow.md`
- Test: `tests/app/test_interface.py`

**Interfaces:**
- Consumes: every earlier task, plus `headline_lines`, `coverage_warnings`, `read_activity`, and the figure writers.
- Produces: `choose_folder() -> str | None` in `frog_classifier.app.dialog`; the Streamlit app with widget keys `view` (sidebar radio), `source`, `folder`, `browse`, `uploads`, `deployment_name`, `scan`, `stop`, `view_results`, `continue`, `new_scan`, `deployment`, `threshold:<record folder>`, `offset:<record folder>`, `look:<record folder>`, `hour:<record folder>`, `day:<record folder>`, `download`.

- [ ] **Step 1: Write the failing tests**

Create `tests/app/test_interface.py`:

```python
from __future__ import annotations

import os
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from frog_classifier.app.paths import ROOT_VARIABLE
from frog_classifier.app.scans import STOP_FILE, ScanStatus, utc_timestamp, write_status
from tests.analytics.helpers import build_record, recording
from tests.inference.helpers import AUDIOMOTH_COMMENT, build_saved_run, write_recording
from tests.preprocessing.helpers import noise


APP = Path(__file__).resolve().parents[2] / "classifier_app" / "app.py"
SCAN_TIMEOUT_S = 300


class InterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        environment = patch.dict(os.environ, {ROOT_VARIABLE: str(self.root)})
        environment.start()
        self.addCleanup(environment.stop)
        self.run_dir = build_saved_run(self.root / "models")
        self.card = self.root / "card"
        for seed, name in enumerate(("20220123_010000.WAV", "20220124_020000.WAV")):
            write_recording(self.card / name, noise(10, sample_rate_hz=48000, seed=seed), comment=AUDIOMOTH_COMMENT)
        self.folder = f"site-a-{self.run_dir.name}"

    def _app(self, view: str | None = None) -> AppTest:
        app = AppTest.from_file(str(APP), default_timeout=60)
        app.run()
        if view is not None:
            app.radio(key="view").set_value(view).run()
        self._assert_no_exception(app)
        return app

    def _assert_no_exception(self, app: AppTest) -> None:
        self.assertEqual(len(app.exception), 0, [exception.value for exception in app.exception])

    def _record(self) -> Path:
        return build_record(self.root / "results" / "detections" / self.folder, [
            recording("20220123_010000.WAV", "2022-01-23T01:00:00", (0.9, 0.2)),
            recording("20220124_020000.WAV", "2022-01-24T02:00:00", (0.97, 0.1)),
            recording("19700101_000000.WAV", "1970-01-01T00:00:00", (0.9,)),
        ], input_root=self.card.as_posix(), model_run=self.run_dir.name)

    def _scanning(self, record: Path) -> None:
        now = utc_timestamp(datetime.now(timezone.utc))
        write_status(record, ScanStatus("running", now, now, False, found=10, processed=4))

    @staticmethod
    def _texts(elements) -> str:
        return "\n".join(element.value for element in elements)

    def test_a_folder_is_previewed_and_suggests_its_own_name(self) -> None:
        app = self._app()

        app.text_input(key="folder").set_value(str(self.card)).run()

        self.assertIn("Found 2 recordings, dated 23 January 2022 to 24 January 2022.", self._texts(app.success))
        self.assertEqual(app.text_input(key="deployment_name").value, "card")
        self.assertFalse(app.button(key="scan").disabled)

    def test_browse_fills_in_the_chosen_folder(self) -> None:
        app = self._app()

        with patch("frog_classifier.app.dialog.choose_folder", return_value=str(self.card)):
            app.button(key="browse").click().run()

        self.assertEqual(app.text_input(key="folder").value, str(self.card))

    def test_the_upload_box_is_offered(self) -> None:
        app = self._app()

        app.radio(key="source").set_value("Upload a few files").run()

        self.assertEqual(len(app.get("file_uploader")), 1)
        self.assertTrue(app.button(key="scan").disabled)

    def test_a_scan_runs_to_the_end_and_opens_its_results(self) -> None:
        app = self._app()
        app.text_input(key="folder").set_value(str(self.card)).run()

        app.button(key="scan").click().run()
        deadline = time.monotonic() + SCAN_TIMEOUT_S
        while "finished" not in self._texts(app.success):
            self._assert_no_exception(app)
            self.assertLess(time.monotonic(), deadline, "the scan did not finish in time")
            time.sleep(1)
            app.run()
        app.button(key="view_results").click().run()

        self.assertEqual(app.radio(key="view").value, "Results")
        self.assertIn("windows of 5 seconds", self._texts(app.markdown))

    def test_a_running_scan_can_be_stopped(self) -> None:
        record = self._record()
        self._scanning(record)
        app = self._app()

        app.button(key="stop").click().run()

        self.assertTrue((record / STOP_FILE).exists())

    def test_results_show_the_summary_warnings_charts_and_download(self) -> None:
        self._record()

        app = self._app("Results")

        self.assertIn(
            "Calls were detected in 50.0% of recorded time (2 of 4 windows of 5 seconds) at threshold 0.500.",
            self._texts(app.markdown),
        )
        self.assertIn("impossible date", self._texts(app.warning))
        self.assertIn("one site", self._texts(app.info))
        self.assertGreaterEqual(len(app.get("imgs")), 3)
        self.assertEqual(len(app.get("download_button")), 1)

    def test_a_new_threshold_changes_the_results(self) -> None:
        self._record()
        app = self._app("Results")

        app.slider(key=f"threshold:{self.folder}").set_value(0.95).run()

        self.assertIn("(1 of 4 windows of 5 seconds) at threshold 0.950.", self._texts(app.markdown))

    def test_the_windows_behind_an_hour_are_listed_and_can_be_heard(self) -> None:
        self._record()

        app = self._app("Results")

        self.assertEqual(app.selectbox(key=f"hour:{self.folder}").value, 1)
        self.assertEqual(len(app.dataframe[0].value), 2)
        self.assertEqual(len(app.get("audio")), 1)

    def test_a_deployment_still_scanning_shows_progress_instead_of_results(self) -> None:
        self._scanning(self._record())

        app = self._app("Results")

        self.assertIn("still being scanned (4 of 10 recordings so far)", self._texts(app.info))
        self.assertEqual(len(app.get("imgs")), 0)


if __name__ == "__main__":
    unittest.main()
```

In `tests/test_repository_reproducibility.py`, add after `ANALYTICS_PACKAGE_MODULES`:

```python
APP_PACKAGE_MODULES = tuple(
    f"src/frog_classifier/app/{name}.py"
    for name in (
        "__init__", "deployments", "dialog", "export", "listening", "models", "paths", "scan_job", "scans", "uploads",
    )
)
```

add `*APP_PACKAGE_MODULES,` and `"classifier_app/app.py",` after `*ANALYTICS_PACKAGE_MODULES,` in the navigation test, add `"classifier_app/app.py",` as the first entry of `OPERATIONAL_TEXT_FILES` after the documents, and add `"uv run streamlit run classifier_app/app.py",` after the `analyze_activity.py` command in the workflow test.

In `tests/app/test_paths_and_models.py`, import `subprocess` and `sys` and add to `AppPathsTests`:

```python
    def test_the_app_modules_do_not_load_pytorch(self) -> None:
        modules = ("deployments", "dialog", "export", "listening", "models", "paths", "scan_job", "scans", "uploads")
        imports = ", ".join(f"frog_classifier.app.{name}" for name in modules)
        result = subprocess.run(
            [sys.executable, "-c", f"import sys, {imports}; print('torch' in sys.modules)"],
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertEqual(result.stdout.strip(), "False")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.app.test_interface tests.test_repository_reproducibility tests.app.test_paths_and_models`
Expected: the interface tests error because `classifier_app/app.py` does not exist; the repository tests fail for the missing files and command; the import test fails for the missing `dialog` module.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/app/dialog.py`:

```python
"""The system folder dialog, for choosing recordings on this computer."""

from __future__ import annotations


def choose_folder() -> str | None:
    """Ask for a folder with the system dialog; None when cancelled or when no dialog can open.

    The app runs on the person's own computer, so the dialog opens on their screen.
    """
    try:
        import tkinter
        from tkinter import filedialog
    except ImportError:
        return None
    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        return None
    root.withdraw()
    # Without this the dialog can open behind the browser window.
    root.attributes("-topmost", True)
    try:
        chosen = filedialog.askdirectory(parent=root, title="Choose the folder or memory card with the recordings")
    finally:
        root.destroy()
    return chosen or None
```

Create `classifier_app/app.py`:

```python
"""Frog call scanner: find Litoria aurea calls in field recordings and see when frogs call.

Start it with: uv run streamlit run classifier_app/app.py
Scans run in their own process, so this page never loads the model itself.
"""

from __future__ import annotations

import io
from datetime import date, datetime, timezone
from pathlib import Path

import streamlit as st

from frog_classifier.analytics.activity import Activity, ActivityBin, ActivityError, PlacedWindow, read_activity
from frog_classifier.analytics.figures import write_by_date_figure, write_clock_figure, write_time_of_day_figure
from frog_classifier.analytics.report import coverage_warnings, headline_lines
from frog_classifier.analytics.timebase import OffsetError
from frog_classifier.app import dialog
from frog_classifier.app.deployments import (
    NAME_LIMIT,
    Deployment,
    DeploymentError,
    RecordingsPreview,
    clean_name,
    list_deployments,
    live_scan,
    preview_recordings,
    start_scan,
)
from frog_classifier.app.export import results_zip
from frog_classifier.app.listening import ListeningError, WindowView, load_window, model_preprocessing
from frog_classifier.app.models import ModelError, ModelInfo, newest_model
from frog_classifier.app.paths import AppPaths, app_paths
from frog_classifier.app.scans import ScanStatus, request_stop
from frog_classifier.app.uploads import UploadError, save_uploads
from frog_classifier.inference.store import DATABASE_FILE, DetectionStore, StoreError


REPO_ROOT = Path(__file__).resolve().parents[1]
VIEW_SCAN = "Scan recordings"
VIEW_RESULTS = "Results"
VIEW_ABOUT = "About"
SOURCE_FOLDER = "A folder or memory card"
SOURCE_UPLOAD = "Upload a few files"
LOOK_HOUR = "An hour of day"
LOOK_DATE = "A date"
FIGURE_TITLES = ("By hour of day", "24-hour clock", "By date")
FIGURE_WRITERS = (write_time_of_day_figure, write_clock_figure, write_by_date_figure)
LISTEN_LIMIT = 100
DEVICES = {"cuda": "Using the graphics card.", "cpu": "Using the processor; a graphics card would be faster."}
VALIDATION_NOTICE = (
    "This detector was trained and checked on recordings from one site, IWS076, made between midnight and noon "
    "from January to April 2022. At other sites, seasons, or hours, listen to a sample of detections before "
    "relying on the numbers."
)
ABOUT = """
This app finds the calls of the green and golden bell frog, *Litoria aurea*, in field recordings.

**Scan recordings** cuts every recording into five-second windows and asks the detector whether each one holds a call.
A whole memory card takes a few minutes.
A scan can be stopped and continued later, and closing the app stops it.

**Results** shows when frogs called:

- *Calls in recorded time* is the share of the recorded five-second windows that hold a call.
- Hours and dates that were never recorded are hatched grey and marked not recorded: nothing is known about them, so they are not zero.
- Every time uses one fixed UTC offset, so daylight-saving changes cannot shift the pattern.
- *Look closer* lists the windows behind any hour or date and plays them, so every number can be checked by ear.
- *Download results* saves the report, charts, and tables as one zip file.
"""


def main() -> None:
    st.set_page_config(page_title="Frog call scanner", page_icon=":frog:", layout="wide")
    paths = app_paths(REPO_ROOT)
    view = st.sidebar.radio("View", (VIEW_SCAN, VIEW_RESULTS, VIEW_ABOUT), key="view")
    if view == VIEW_SCAN:
        scan_view(paths)
    elif view == VIEW_RESULTS:
        results_view(paths)
    else:
        about_view(paths)


# Scanning


def scan_view(paths: AppPaths) -> None:
    st.title("Scan recordings")
    try:
        model = newest_model(paths.models)
    except ModelError as error:
        st.error(f"Scanning is not possible yet: {error}.")
        return
    problem = st.session_state.pop("scan_problem", None)
    if problem is not None:
        st.error(f"Cannot scan: {problem}.")
    deployments = list_deployments(paths.detections, now=_now())
    live = live_scan(deployments)
    if live is not None:
        st.session_state["scan_record"] = live.record_dir.name
    chosen = next((d for d in deployments if d.record_dir.name == st.session_state.get("scan_record")), None)
    if chosen is None:
        _scan_form(paths, model)
    elif chosen.scanning:
        _live_progress(chosen.record_dir.name, chosen.name, paths)
    else:
        _scan_outcome(paths, model, chosen)


def _scan_form(paths: AppPaths, model: ModelInfo) -> None:
    st.write("Choose the recordings from one recorder deployment or memory card, give them a name, and press Scan.")
    source = st.radio("Where are the recordings?", (SOURCE_FOLDER, SOURCE_UPLOAD), horizontal=True, key="source")
    if source == SOURCE_FOLDER:
        ready = _folder_choice()
    else:
        uploads = st.file_uploader(
            "Recordings",
            type=["wav", "mp3"],
            accept_multiple_files=True,
            key="uploads",
            help="Uploading copies each file into the app. For a whole memory card, choose its folder instead.",
        )
        ready = bool(uploads)
    st.text_input(
        "Name for these recordings",
        key="deployment_name",
        max_chars=NAME_LIMIT,
        help="For example the site and card number, such as IWS076 card 3. Results are kept under this name.",
    )
    st.button("Scan", key="scan", type="primary", disabled=not ready, on_click=_start_from_form, args=(paths, model))
    st.caption(f"Detector {model.name}, threshold {model.threshold:.3f}.")


def _folder_choice() -> bool:
    folder_column, browse_column = st.columns([5, 1], vertical_alignment="bottom")
    folder_column.text_input("Folder", key="folder", placeholder=r"For example E:\ for a memory card, or D:\recordings\site-a")
    browse_column.button("Browse...", key="browse", on_click=_browse)
    text = _folder_text()
    if not text:
        st.info("Paste the folder's location, or press Browse to pick it. A memory card usually appears as a drive such as E:\\.")
        return False
    try:
        preview = preview_recordings(Path(text))
    except DeploymentError as error:
        st.error(f"{str(error).capitalize()}.")
        return False
    if preview.count == 0:
        st.warning("There are no .wav or .mp3 recordings in this folder or its subfolders.")
        return False
    st.success(_preview_text(preview))
    if st.session_state.get("named_folder") != text:
        # A new folder suggests its own name; a name typed for this folder is kept.
        st.session_state["named_folder"] = text
        st.session_state["deployment_name"] = Path(text).resolve().name
    return True


def _preview_text(preview: RecordingsPreview) -> str:
    text = f"Found {preview.count:,} recording{'s' if preview.count != 1 else ''}"
    if preview.first_day is not None:
        text += f", dated {_day(preview.first_day)}"
        if preview.last_day != preview.first_day:
            text += f" to {_day(preview.last_day)}"
    text += "."
    if preview.undated:
        text += f" {preview.undated:,} {'has' if preview.undated == 1 else 'have'} no date in the file name."
    return text


@st.fragment(run_every=1.0)
def _live_progress(folder: str, name: str, paths: AppPaths) -> None:
    deployment = next((d for d in list_deployments(paths.detections, now=_now()) if d.record_dir.name == folder), None)
    if deployment is None or not deployment.scanning:
        st.rerun()
    status = deployment.status
    st.subheader(f"Scanning {name}")
    if status.state == "starting":
        st.progress(0.0, text="Starting: loading the detector...")
    else:
        fraction = status.done / status.found if status.found else 1.0
        st.progress(fraction, text=f"Scanned {status.done:,} of {status.found:,} recordings{_time_left(status)}")
    if status.failed:
        st.warning(f"{status.failed:,} recordings could not be read so far.")
    if status.device is not None:
        st.caption(DEVICES.get(status.device, status.device))
    st.button(
        "Stop scan",
        key="stop",
        on_click=request_stop,
        args=(deployment.record_dir,),
        help="Finished recordings are kept, and the scan can be continued later.",
    )


def _scan_outcome(paths: AppPaths, model: ModelInfo, deployment: Deployment) -> None:
    status = deployment.status
    st.subheader(deployment.name)
    if status is None or status.state == "finished":
        counted = f": {status.found:,} recordings" if status is not None else ""
        st.success(f"The scan of {deployment.name} finished{counted}.")
        if status is not None and status.failed:
            st.warning(f"{status.failed:,} recordings could not be read. Continue scan tries them again.")
        if status is not None and status.message:
            st.warning(status.message)
    elif status.state in ("stopped", "interrupted"):
        how = "was stopped" if status.state == "stopped" else "ended unexpectedly, for example because the app was closed,"
        progress = f"after {status.done:,} of {status.found:,} recordings" if status.found else "before it began scanning"
        st.warning(
            f"The scan of {deployment.name} {how} {progress}. "
            "Finished recordings are kept, and continuing picks up where it stopped."
        )
    else:
        st.error(f"The scan of {deployment.name} could not run: {status.message}")
    columns = st.columns(3)
    if deployment.results_ready:
        columns[0].button(
            "View results", key="view_results", type="primary", on_click=_open_results, args=(deployment.record_dir.name,),
        )
    if status is not None and (status.state != "finished" or status.failed):
        columns[1].button(
            "Continue scan",
            key="continue",
            on_click=_start,
            args=(paths, model, Path(deployment.input_root), deployment.name),
        )
    columns[2].button("Scan other recordings", key="new_scan", on_click=_forget_scan)


def _time_left(status: ScanStatus) -> str:
    if status.processed < 3:
        return ""
    elapsed = (_now() - datetime.fromisoformat(status.started_at)).total_seconds()
    seconds = elapsed / status.processed * (status.found - status.done)
    if seconds < 60:
        return ", less than a minute left"
    minutes = round(seconds / 60)
    return f", about {minutes} minute{'s' if minutes != 1 else ''} left"


def _start_from_form(paths: AppPaths, model: ModelInfo) -> None:
    name = st.session_state.get("deployment_name", "")
    try:
        if st.session_state.get("source") == SOURCE_UPLOAD:
            folder = paths.uploads / clean_name(name)
            uploads = st.session_state.get("uploads") or []
            save_uploads([(upload.name, upload.getvalue()) for upload in uploads], folder)
        else:
            folder = Path(_folder_text())
    except (DeploymentError, UploadError) as error:
        st.session_state["scan_problem"] = str(error)
        return
    _start(paths, model, folder, name)


def _start(paths: AppPaths, model: ModelInfo, folder: Path, name: str) -> None:
    try:
        record_dir = start_scan(paths, model, folder, name, now=_now())
    except DeploymentError as error:
        st.session_state["scan_problem"] = str(error)
        return
    st.session_state["scan_record"] = record_dir.name


def _browse() -> None:
    chosen = dialog.choose_folder()
    if chosen:
        st.session_state["folder"] = str(Path(chosen))


def _folder_text() -> str:
    return st.session_state.get("folder", "").strip().strip('"')


def _forget_scan() -> None:
    st.session_state.pop("scan_record", None)


def _open_results(folder: str) -> None:
    st.session_state["view"] = VIEW_RESULTS
    st.session_state["deployment"] = folder


def _open_scan(folder: str) -> None:
    st.session_state["view"] = VIEW_SCAN
    st.session_state["scan_record"] = folder


# Results


def results_view(paths: AppPaths) -> None:
    st.title("Results")
    deployments = list_deployments(paths.detections, now=_now())
    if not deployments:
        st.info("Nothing has been scanned yet. Start with Scan recordings.")
        return
    by_folder = {deployment.record_dir.name: deployment for deployment in deployments}
    if st.session_state.get("deployment") not in by_folder:
        st.session_state["deployment"] = deployments[0].record_dir.name
    names = [deployment.name for deployment in deployments]
    folder = st.selectbox(
        "Recordings",
        list(by_folder),
        key="deployment",
        format_func=lambda folder: _deployment_label(by_folder[folder], names),
    )
    deployment = by_folder[folder]
    if deployment.scanning:
        status = deployment.status
        st.info(
            f"{deployment.name} is still being scanned ({status.done:,} of {status.found:,} recordings so far). "
            "Its results appear here when the scan finishes."
        )
        return
    if not deployment.results_ready:
        st.warning(
            f"The scan of {deployment.name} did not finish, so its results would be incomplete. "
            "Continue it under Scan recordings."
        )
        st.button("Go to Scan recordings", key="go_scan", on_click=_open_scan, args=(folder,))
        return
    _results(paths, deployment)


def _results(paths: AppPaths, deployment: Deployment) -> None:
    folder = deployment.record_dir.name
    database = deployment.record_dir / DATABASE_FILE
    try:
        with DetectionStore.open_read_only(database) as store:
            identity = store.identity
        stamp = database.stat().st_mtime_ns
    except (OSError, StoreError) as error:
        st.error(f"The results of {deployment.name} cannot be read: {error}.")
        return
    with st.expander("Threshold and time zone"):
        threshold = st.slider(
            "Detection threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(identity.threshold),
            step=0.001,
            format="%.3f",
            key=f"threshold:{folder}",
            help="A window counts as a call when the detector's probability is at or above this. "
            "The default is the detector's own threshold.",
        )
        offset = st.text_input(
            "UTC offset",
            key=f"offset:{folder}",
            placeholder="The recorders' own, such as +11:00",
            help="One fixed UTC offset for every time, such as +10:00 for Australian Eastern Standard Time.",
        ).strip() or None
    try:
        activity, figures = _analysis(str(deployment.record_dir), stamp, threshold, offset)
    except (StoreError, OffsetError, ActivityError) as error:
        st.error(f"The results of {deployment.name} cannot be shown: {error}.")
        return
    st.subheader(deployment.name)
    st.markdown("\n".join(f"- {line}" for line in headline_lines(activity)))
    for warning in coverage_warnings(activity):
        st.warning(warning)
    st.caption(
        f"Detector {identity.model_run} at threshold {activity.threshold:.3f}; its own threshold is "
        f"{identity.threshold:.3f}. Times are in {activity.time_basis.label}. Recordings from {identity.input_root}."
    )
    st.info(VALIDATION_NOTICE)
    for tab, figure in zip(st.tabs(FIGURE_TITLES), figures):
        tab.image(figure)
    st.markdown("#### How much the threshold matters")
    st.table([
        {
            "Threshold": f"{row.threshold:.3f}",
            "Windows with calls": f"{row.detected_windows:,}",
            "Calls in recorded time": "not recorded" if row.calling_share is None else f"{row.calling_share:.1%}",
        }
        for row in activity.thresholds
    ])
    _look_closer(paths, deployment, activity)
    st.download_button(
        "Download results (zip)",
        key="download",
        type="primary",
        file_name=f"{folder}.zip",
        mime="application/zip",
        data=lambda: results_zip(deployment.record_dir, threshold=threshold, utc_offset=offset),
    )


def _look_closer(paths: AppPaths, deployment: Deployment, activity: Activity) -> None:
    folder = deployment.record_dir.name
    st.markdown("#### Look closer")
    st.caption("Pick an hour of day or a date to see every five-second window behind its bar, and listen to any of them.")
    mode = st.radio("Look at", (LOOK_HOUR, LOOK_DATE), horizontal=True, key=f"look:{folder}")
    if mode == LOOK_HOUR:
        hours = [hour for hour, bin_ in enumerate(activity.by_hour) if bin_.recorded_windows]
        if not hours:
            st.info("No window could be placed in time.")
            return
        hour = st.selectbox(
            "Hour of day",
            hours,
            key=f"hour:{folder}",
            format_func=lambda hour: f"{hour:02d}:00 to {hour:02d}:59, {_bin_text(activity.by_hour[hour])}",
        )
        windows, chosen = [window for window in activity.windows if window.hour == hour], f"{hour:02d}"
    else:
        days = [day for day, bin_ in activity.by_date.items() if bin_.recorded_windows]
        day = st.selectbox(
            "Date",
            days,
            key=f"day:{folder}",
            format_func=lambda day: f"{_day(day)}, {_bin_text(activity.by_date[day])}",
        )
        windows, chosen = [window for window in activity.windows if window.date == day], day.isoformat()
    windows.sort(key=lambda window: (not activity.is_detected(window), -window.probability, window.start))
    st.dataframe(
        [
            {
                "Date": f"{window.start:%Y-%m-%d}",
                "Time": f"{window.start:%H:%M:%S}",
                "Call": "yes" if activity.is_detected(window) else "",
                "Probability": round(window.probability, 3),
                "Recording": window.recording_path,
                "Seconds in": window.window_start_s,
            }
            for window in windows
        ],
        hide_index=True,
        height=240,
    )
    index = st.selectbox(
        "Listen to",
        list(range(min(len(windows), LISTEN_LIMIT))),
        key=f"listen:{folder}:{chosen}",
        format_func=lambda index: _window_label(windows[index], activity),
        help=f"The first {LISTEN_LIMIT} windows in the table, calls first.",
    )
    window = windows[index]
    try:
        view = _window_view(
            deployment.input_root, window.recording_path, window.window_start_s, str(paths.models), deployment.model_run,
        )
    except ListeningError as error:
        st.warning(f"{str(error).capitalize()}.")
        return
    sound_column, image_column = st.columns([1, 2])
    sound_column.audio(view.wav, format="audio/wav")
    sound_column.caption("Played at full loudness, so faint calls can be heard.")
    image_column.image(
        view.image,
        caption="The spectrogram the detector saw: low pitches at the bottom, five seconds from left to right.",
        width="stretch",
    )


@st.cache_data(max_entries=16, show_spinner="Reading the results...")
def _analysis(record_dir: str, stamp: int, threshold: float, utc_offset: str | None) -> tuple[Activity, list[bytes]]:
    """The activity and its three figures; stamp, the database's modification time,
    makes a changed record a new cache entry."""
    with DetectionStore.open_read_only(Path(record_dir) / DATABASE_FILE) as store:
        activity = read_activity(store, threshold=threshold, utc_offset=utc_offset)
    figures = []
    for write in FIGURE_WRITERS:
        buffer = io.BytesIO()
        write(buffer, activity)
        figures.append(buffer.getvalue())
    return activity, figures


@st.cache_data(max_entries=32, show_spinner="Loading the recording...")
def _window_view(input_root: str, recording_path: str, start_s: int, models_root: str, model_run: str) -> WindowView:
    config = model_preprocessing(Path(models_root), model_run)
    return load_window(Path(input_root) / recording_path, start_s, config)


def _deployment_label(deployment: Deployment, names: list[str]) -> str:
    label = deployment.name
    if names.count(deployment.name) > 1:
        label += f" ({deployment.model_run})"
    if deployment.scanning:
        label += " (scanning)"
    elif not deployment.results_ready:
        label += " (not finished)"
    return label


def _day(day: date) -> str:
    """A date as people write it, such as 3 April 2022."""
    return f"{day.day} {day:%B %Y}"


def _bin_text(bin_: ActivityBin) -> str:
    return f"calls in {bin_.calling_share:.1%} of {bin_.recorded_windows:,} windows"


def _window_label(window: PlacedWindow, activity: Activity) -> str:
    call = ", call" if activity.is_detected(window) else ""
    return f"{window.start:%d %b %Y %H:%M:%S}, probability {window.probability:.3f}{call}"


# About


def about_view(paths: AppPaths) -> None:
    st.title("About")
    st.markdown(ABOUT)
    try:
        model = newest_model(paths.models)
    except ModelError as error:
        st.warning(f"{str(error).capitalize()}.")
    else:
        st.write(f"Detector in use: {model.name}, threshold {model.threshold:.3f}.")
    st.info(VALIDATION_NOTICE)


def _now() -> datetime:
    return datetime.now(timezone.utc)


main()
```

Delete `classifier_app/.gitkeep`.

In `docs/workflow.md`, add this section after "Analyze calling activity":

````markdown
## Use the app

The app does scanning and analysis without a command line.
Start it from the repository folder; it opens in the browser at `http://localhost:8501`:

```powershell
uv run streamlit run classifier_app/app.py
```

Under **Scan recordings**, paste a folder or memory card location or press Browse, check how many recordings it found and their dates, name the deployment, and press Scan.
A few recordings can be uploaded instead; they are saved under `raw/uploads/<name>/` and never overwritten.
The scan runs in its own process and shows its progress, can be stopped, and continues where it stopped; closing the app's console window also stops it.
Under **Results**, pick a deployment to see the summary, warnings, the three charts, and the threshold table, change the threshold or the UTC offset, list the windows behind any hour or date and listen to them, and download everything as one zip file.
Results appear only once a deployment's scan has finished.

Each deployment's record folder is `results/detections/<name>-<model run>/`.
Beside the detection files it holds `deployment.json`, the name and source folder; `scan_status.json`, the scan's progress; and `scan.log`, the scan process's output.
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.app.test_interface tests.test_repository_reproducibility tests.app.test_paths_and_models`
Expected: every test passes.

- [ ] **Step 5: Commit**

```bash
git rm -q classifier_app/.gitkeep
git add src/frog_classifier/app/dialog.py classifier_app/app.py tests/app/test_interface.py tests/test_repository_reproducibility.py docs/workflow.md
git commit -m "feat: add the site staff app"
```

---

### Task 10: Real-data acceptance

**Files:**
- No tracked files change, except fixes found here, each with a test.

- [ ] **Step 1: Full verification**

Run the full suite to a log file, `compileall` over `scripts`, `src`, and `classifier_app`, `uv lock --check`, and `git diff --check`.
Expected: the suite reports `OK`.

- [ ] **Step 2: Results of the existing record in the app**

Start the app on the repository data with the in-app browser and open Results for `external`.
Expected: the summary reads 618 recordings on 58 of 76 dates, 21.8% at threshold 0.794, peak 01:00 at 34.2%, no warnings, and the charts match part 1; a detection plays and shows its spectrogram.

- [ ] **Step 3: A real scan through the app**

Start a second app instance whose `FROG_CLASSIFIER_ROOT` is a scratch folder holding a copy of `models/resnet18-20261014T114432Z`, scan `raw/external` as a new deployment, stop it partway, continue it, and let it finish.
Expected: progress moves, the stop leaves whole recordings, the continuation finishes, and its `activity_by_hour.csv` and `activity_by_date.csv` from `results_zip` equal those of the command-line analysis.

- [ ] **Step 4: The download**

Build `results_zip` for the real record and list it.
Expected: one folder with the report, three figures, three tables, and both detection spreadsheets.

---

### Task 11: Documentation and status

**Files:**
- Modify: `docs/architecture.md`, `docs/outline.md`, `README.md`, `roadmap.md`, `AGENTS.md`, `docs/workflow.md`

- [ ] **Step 1: Update the documents**

- `docs/architecture.md`: a "Site staff app" section with the module table, the two processes, the status file and `results_ready`, stopping, deployments and `deployment.json`, and impossible start times in the activity section; storage lines for `raw/uploads/` and `src/frog_classifier/app/`; the pipeline diagram.
- `docs/workflow.md`: the `time` column of `windows.csv`, the impossible-date rule, and the measured results of Task 10.
- `docs/outline.md` and `README.md`: the app as the way site staff use the project, with its command.
- `roadmap.md`: Phase 6 part 2 done and verified, part 3 next.
- `AGENTS.md`: part 2 of Phase 6 done.

- [ ] **Step 2: Verify and commit**

Run the full suite, `git diff --check`, and the em dash scan over every changed file, then commit the documents.

---

## Execution record

Executed inline on `main` on October 15, 2026, with a final self-review instead of a separate reviewer.

Rulings made during execution:

- `test_figures_can_be_drawn_into_memory` passed before its change, as the plan predicted, because matplotlib already accepted a stream; the test pins the app's reliance on it.
- The browser check found three display defects: the drive hint lost the backslash in `E:\` to Markdown, the threshold table showed meaningless row numbers, and probabilities read 1 beside 0.002; each was fixed with an interface test.
- Streamlit's developer toolbar, with its Deploy button, still shows; part 3, packaging, owns how the app is started and will hide it and turn off Streamlit's usage statistics.
- The scan process writes "finished" a moment before it exits, and Windows will not delete its open `scan.log`, so the end-to-end interface test once failed in cleanup; it now waits for the launched process and asserts that it exits with 0.
- Streamlit's test harness cannot simulate an upload or a download, so saving uploads and building the zip are tested directly, and the roadmap says so.
- The final review found that Look closer failed when no recording had a usable date, as on a card whose clock was never set, and the date view was chosen; it now shows a message and no controls, with a test.
- The final review found that the documented command let Streamlit listen on every network interface, so anyone on the same network could list folders and play recordings; every documented command now adds `--server.address localhost`, as the labeler launcher does, with a test, and the app then listens on 127.0.0.1 and ::1 only.
- The download's `detections.csv` lists windows at the model's threshold even when another threshold is chosen; every row states its threshold, and the report and `windows.csv` use the chosen one, so it stays as it is.

Deferred minor items:

- A memory card mounted at a different drive letter cannot continue its earlier scan, because the record's identity holds the absolute folder; scanning it under a new name works.
- The folder preview walks the whole folder on every page interaction, so pasting a whole drive such as `D:\` is slow.
- Uploads are saved before the name is checked, so a refused start can leave copies in `raw/uploads/<name>/`.
- The Scan button is enabled with an empty name, such as after choosing a card's root; the error explains, but disabling it would be clearer.
- A scan process that fails before its first status reads as "ended unexpectedly" without pointing to `scan.log`.
- A recording that became unreadable after its scan raises an unhandled decode error when listening.
- The Listen to choice keeps its position when the threshold changes, so it can point at another window; its label shows which.
- The comment on `scans._launched` does not say that it keeps process handles alive so Python does not warn about a running child.
- Browse was checked on Windows only; macOS needs Tk on the main thread.
