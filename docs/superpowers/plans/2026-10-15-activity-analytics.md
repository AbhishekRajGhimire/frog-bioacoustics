# Activity Analytics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a detection record into calling activity by hour of day and by date, with recording effort kept separate and every value traceable to windows.

**Architecture:** A new `frog_classifier.analytics` package reads a detection record read-only, places every window in one fixed UTC offset, aggregates effort and detections by hour and by date, draws three figures, and writes a report and tables. A shared `frog_classifier.files.replace_file` writes every generated file beside its target and swaps it in, for both analytics and the detection export. `scripts/analyze_activity.py` wraps it.

**Tech Stack:** Python 3.13 standard library (`sqlite3`, `datetime`), matplotlib, unittest.

**Spec:** `docs/superpowers/specs/2026-10-15-activity-analytics-design.md`

## Global Constraints

- No new dependencies: `pyproject.toml`, `uv.lock`, and `requirements_labeler.txt` do not change.
- uv is at `/c/Users/ghimi/.local/bin/uv.exe`; it is not on PATH in this shell. Run Python as `.venv/Scripts/python.exe`.
- Text files use LF line endings; write files with the editor tools or bytes, never Python `write_text` on Windows.
- Never use the em dash character anywhere. Each full sentence in long Markdown files sits on its own line.
- Commit messages carry no co-author trailer.
- Tests build synthetic detection records in temporary folders and never read `raw/`, `labeled/`, `models/`, or `results/`.
- `frog_classifier.analytics` never imports PyTorch.
- Analysis never writes to a detection record.
- matplotlib is used through `matplotlib.figure.Figure`, never `pyplot`, and every `savefig` names `format="png"` because files are first written under a `.partial` name.

## Review Focus

- A recording that crosses midnight or the top of an hour contributes to both bins (Task 4 test).
- Hours and dates with no recording read as not recorded, never as zero calls (Tasks 4 and 6 tests).
- A record whose detection run is still writing or was interrupted is refused with advice instead of a raw SQLite error (Task 2 test).
- A recording set with mixed stated offsets is placed in one offset (Task 4 test).
- A record where nothing can be placed in time still produces figures and a report (Tasks 5 and 6 tests).

---

### Task 1: Shared write-then-swap for generated files

**Files:**
- Create: `src/frog_classifier/files.py`
- Modify: `src/frog_classifier/inference/export.py`
- Modify: `src/frog_classifier/inference/detect.py`
- Test: `tests/test_files.py`, `tests/inference/test_export.py`, `tests/inference/test_detect.py`

**Interfaces:**
- Produces: `FileInUseError(Exception)` and `replace_file(path: Path, write: Callable[[Path], None]) -> None` in `frog_classifier.files`. `ExportError` is removed; `export_csv` raises `FileInUseError`, and `detect_calls` turns it into `DetectionSummary.csv_problem`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_files.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frog_classifier.files import FileInUseError, replace_file


class ReplaceFileTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.target = self.root / "report.md"
        self.target.write_bytes(b"old")

    def test_the_new_content_replaces_the_old(self) -> None:
        replace_file(self.target, lambda staging: staging.write_bytes(b"new"))

        self.assertEqual(self.target.read_bytes(), b"new")
        self.assertEqual([path.name for path in self.root.iterdir()], ["report.md"])

    def test_a_file_that_cannot_be_replaced_keeps_its_content_and_explains_why(self) -> None:
        with patch("frog_classifier.files.os.replace", side_effect=PermissionError(13, "Permission denied")):
            with self.assertRaisesRegex(
                FileInUseError, "report.md could not be replaced.*close it and run the same command again"
            ):
                replace_file(self.target, lambda staging: staging.write_bytes(b"new"))

        self.assertEqual(self.target.read_bytes(), b"old")
        self.assertEqual([path.name for path in self.root.iterdir()], ["report.md"])


if __name__ == "__main__":
    unittest.main()
```

In `tests/inference/test_export.py`, import `FileInUseError` from `frog_classifier.files` instead of `ExportError` from the export module, and use it in `test_a_csv_that_cannot_be_replaced_is_explained_and_leaves_no_partial_file`, whose patch target becomes `frog_classifier.files.os.replace`.

Add to `DetectCallsTests` in `tests/inference/test_detect.py` (add `from unittest.mock import patch` to the imports):

```python
    def test_a_spreadsheet_that_cannot_be_replaced_is_reported_and_the_record_is_kept(self) -> None:
        with patch("frog_classifier.files.os.replace", side_effect=PermissionError(13, "Permission denied")):
            summary = self._detect()

        self.assertIn("detections.csv could not be replaced", summary.csv_problem)
        self.assertIn("The detection database is complete", summary.csv_problem)
        self.assertEqual(self._query("SELECT COUNT(*) FROM recordings"), [(4,)])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.test_files tests.inference.test_export tests.inference.test_detect`
Expected: ERROR with `ModuleNotFoundError: No module named 'frog_classifier.files'`.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/files.py`:

```python
"""Replace generated files without ever leaving half a file behind."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


class FileInUseError(Exception):
    pass


def replace_file(path: Path, write: Callable[[Path], None]) -> None:
    """Call write with a temporary path beside path, then swap the result into place.

    A reader sees either the old file or the new one. If path is open in another
    program and cannot be replaced, the temporary file is removed and the old
    file stays as it was.
    """
    staging = path.with_name(f".{path.name}.partial")
    write(staging)
    try:
        os.replace(staging, path)
    except PermissionError as error:
        staging.unlink()
        raise FileInUseError(
            f"{path.name} could not be replaced, probably because it is open in another program; "
            "close it and run the same command again"
        ) from error
```

In `src/frog_classifier/inference/export.py`, remove `import os` and `ExportError`, import `replace_file` from `frog_classifier.files`, and replace `_replace` with:

```python
def _replace(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    replace_file(path, lambda staging: write_csv(staging, columns, rows))
```

In `src/frog_classifier/inference/detect.py`, import `FileInUseError` from `frog_classifier.files`, import only `export_csv` from `.export`, and replace the export handling with:

```python
        csv_problem = None
        try:
            export_csv(store, output_dir)
        except FileInUseError as error:
            csv_problem = f"{error}. The detection database is complete, so running it again only rewrites the CSV files."
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.test_files tests.inference.test_export tests.inference.test_detect tests.inference.test_command`
Expected: every test passes.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/files.py src/frog_classifier/inference/export.py src/frog_classifier/inference/detect.py tests/test_files.py tests/inference/test_export.py tests/inference/test_detect.py
git commit -m "refactor: share write-then-swap for generated files"
```

---

### Task 2: Read-only access to a detection record

**Files:**
- Modify: `src/frog_classifier/inference/store.py`
- Test: `tests/inference/test_store.py`

**Interfaces:**
- Produces: `DetectionStore.open_read_only(path: Path) -> DetectionStore`, whose `identity` and `created_at` come from the database; writing through it raises `sqlite3.OperationalError`.

- [ ] **Step 1: Write the failing tests**

Add to `DetectionStoreTests` in `tests/inference/test_store.py`:

```python
    def test_a_read_only_store_takes_its_identity_from_the_database_and_cannot_write(self) -> None:
        store = self._open()
        store.save(recording_row("site/a.wav"), (0, 5), (0.25, 0.75))
        store.close()
        before = self.path.read_bytes()

        with DetectionStore.open_read_only(self.path) as reader:
            self.assertEqual(reader.identity, SCAN_IDENTITY)
            self.assertEqual(reader.created_at, "2026-10-15T00:00:00Z")
            self.assertEqual(reader.recording_rows(), [recording_row("site/a.wav")])
            with self.assertRaises(sqlite3.OperationalError):
                reader.save(recording_row("site/b.wav"), (0, 5), (0.1, 0.2))

        self.assertEqual(self.path.read_bytes(), before)

    def test_reading_a_missing_database_says_to_run_detection_first(self) -> None:
        with self.assertRaisesRegex(StoreError, "run the detection command first"):
            DetectionStore.open_read_only(self.path)

    def test_reading_a_database_that_is_mid_write_is_refused(self) -> None:
        self._open().close()
        self.path.with_name(f"{self.path.name}-journal").write_bytes(b"\0" * 512)

        with self.assertRaisesRegex(StoreError, "still writing .* or was interrupted"):
            DetectionStore.open_read_only(self.path)

    def test_reading_a_file_that_is_not_a_database_is_refused(self) -> None:
        self.path.write_bytes(b"these bytes are not a database " * 40)

        with self.assertRaisesRegex(StoreError, "is not a detection database"):
            DetectionStore.open_read_only(self.path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.inference.test_store`
Expected: four errors with `AttributeError: type object 'DetectionStore' has no attribute 'open_read_only'`.

- [ ] **Step 3: Implement**

In `src/frog_classifier/inference/store.py`, add to `DetectionStore` after `open`:

```python
    @classmethod
    def open_read_only(cls, path: Path) -> DetectionStore:
        """Open an existing detection database for reading only, taking its identity from the database."""
        if not path.is_file():
            raise StoreError(f"{path} does not exist; run the detection command first")
        journal = path.with_name(f"{path.name}-journal")
        if journal.is_file() and journal.stat().st_size > 0:
            raise StoreError(
                f"a detection run is still writing {path} or was interrupted; "
                "let it finish, or run the same detection command again, before reading it"
            )
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            if "scan" not in _table_names(connection, path):
                raise StoreError(f"{path} is not a detection database")
            identity, created_at = _read_scan(connection, path)
        except BaseException:
            connection.close()
            raise
        return cls(connection, identity, created_at)
```

Replace `_check_scan` with `_read_scan`, which states facts without advice, because a reader and a writer need different advice:

```python
def _read_scan(connection: sqlite3.Connection, path: Path) -> tuple[ScanIdentity, str]:
    (version,) = connection.execute("SELECT schema_version FROM scan").fetchone()
    if version != SCHEMA_VERSION:
        raise StoreError(f"{path} uses detection schema {version}, but this code reads schema {SCHEMA_VERSION}")
    row = connection.execute(f"SELECT created_at, {', '.join(_IDENTITY_COLUMNS)} FROM scan").fetchone()
    return ScanIdentity(*row[1:]), row[0]
```

In `_table_names`, the error message becomes `f"{path} is not a detection database ({error})"`.

`open` checks the identity itself and adds the writer's advice to every refusal in one place:

```python
        connection = sqlite3.connect(path, isolation_level=None)
        try:
            tables = _table_names(connection, path)
            connection.execute("PRAGMA foreign_keys = ON")
            if not tables:
                created_at = _create(connection, identity, now)
            elif "scan" not in tables:
                raise StoreError(f"{path} is not a detection database")
            else:
                stored, created_at = _read_scan(connection, path)
                changed = [name for name in _IDENTITY_COLUMNS if getattr(stored, name) != getattr(identity, name)]
                if changed:
                    raise StoreError(f"{path.parent} holds results for a different {', '.join(changed)}")
        except StoreError as error:
            connection.close()
            raise StoreError(f"{error}; choose another output folder") from error
        except BaseException:
            connection.close()
            raise
        return cls(connection, identity, created_at)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.inference.test_store tests.inference.test_detect`
Expected: every test passes.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/inference/store.py tests/inference/test_store.py
git commit -m "feat: open a detection record read-only"
```

---

### Task 3: The analysis time basis

**Files:**
- Create: `src/frog_classifier/analytics/__init__.py`
- Create: `src/frog_classifier/analytics/timebase.py`
- Create: `tests/analytics/__init__.py` (empty)
- Test: `tests/analytics/test_timebase.py`

**Interfaces:**
- Produces: `OffsetError(ValueError)`, `parse_offset(text: str) -> timezone`, `format_offset(zone: timezone) -> str`, `most_common_offset(offsets: Iterable[str | None]) -> str | None`, `TimeBasis(zone: timezone, offset: str | None)` with property `label: str`, `choose_time_basis(requested: str | None, recording_offsets: Iterable[str | None]) -> TimeBasis`, `window_time(started_at: str, recording_offset: str | None, start_s: int, analysis_zone: timezone) -> datetime`.

- [ ] **Step 1: Write the failing tests**

Create `tests/analytics/__init__.py` as an empty file and `tests/analytics/test_timebase.py`:

```python
from __future__ import annotations

import unittest
from datetime import date

from frog_classifier.analytics.timebase import (
    OffsetError,
    choose_time_basis,
    format_offset,
    most_common_offset,
    parse_offset,
    window_time,
)


class OffsetTests(unittest.TestCase):
    def test_offsets_parse_and_format_back_unchanged(self) -> None:
        for text in ("+11:00", "-03:30", "+00:00", "+05:45"):
            with self.subTest(offset=text):
                self.assertEqual(format_offset(parse_offset(text)), text)

    def test_malformed_offsets_are_rejected(self) -> None:
        for text in ("11:00", "+11", "+1100", "UTC+11", "+15:00", "+10:60"):
            with self.subTest(offset=text):
                with self.assertRaises(OffsetError):
                    parse_offset(text)

    def test_the_offset_most_recordings_state_is_chosen(self) -> None:
        self.assertEqual(most_common_offset(["+11:00", "+10:00", "+11:00", None]), "+11:00")
        self.assertEqual(most_common_offset(["+11:00", "+10:00"]), "+10:00")
        self.assertIsNone(most_common_offset([None, None]))


class TimeBasisTests(unittest.TestCase):
    def test_a_requested_offset_wins(self) -> None:
        basis = choose_time_basis("+10:00", ["+11:00"])

        self.assertEqual(basis.label, "UTC+10:00")
        self.assertEqual(format_offset(basis.zone), "+10:00")

    def test_the_recorders_offset_is_the_default(self) -> None:
        self.assertEqual(choose_time_basis(None, ["+11:00", "+11:00", "+10:00"]).label, "UTC+11:00")

    def test_without_any_stated_offset_times_stay_on_the_recorder_clock(self) -> None:
        basis = choose_time_basis(None, [None])

        moment = window_time("2022-01-23T01:00:00", None, 10, basis.zone)

        self.assertEqual(basis.label, "recorder clock, UTC offset unknown")
        self.assertEqual((moment.hour, moment.minute, moment.second), (1, 0, 10))


class WindowTimeTests(unittest.TestCase):
    def test_a_window_is_converted_from_its_recording_offset(self) -> None:
        moment = window_time("2022-04-04T01:00:00", "+11:00", 5, parse_offset("+10:00"))

        self.assertEqual((moment.date(), moment.hour, moment.minute, moment.second), (date(2022, 4, 4), 0, 0, 5))

    def test_a_window_can_fall_on_the_next_date(self) -> None:
        moment = window_time("2022-01-23T23:59:55", "+11:00", 5, parse_offset("+11:00"))

        self.assertEqual((moment.date(), moment.hour), (date(2022, 1, 24), 0))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.analytics.test_timebase`
Expected: ERROR with `ModuleNotFoundError: No module named 'frog_classifier.analytics'`.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/analytics/__init__.py`:

```python
"""Activity analytics: recording effort and detected calls by hour of day and by date.

Nothing is re-exported here; import modules directly. The package never imports
PyTorch, so an application can show analyses without loading the model.
"""
```

Create `src/frog_classifier/analytics/timebase.py`:

```python
"""The analysis time basis: one fixed UTC offset for every window.

Frogs do not follow daylight saving, so times are never converted to a wall-clock
time zone, which would shift activity by an hour at every change.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable


_OFFSET = re.compile(r"(?P<sign>[+-])(?P<hours>\d{2}):(?P<minutes>\d{2})")
# Every UTC offset in use lies between -12:00 and +14:00.
_LARGEST_OFFSET = timedelta(hours=14)


class OffsetError(ValueError):
    pass


@dataclass(frozen=True)
class TimeBasis:
    zone: timezone
    offset: str | None

    @property
    def label(self) -> str:
        if self.offset is None:
            return "recorder clock, UTC offset unknown"
        return f"UTC{self.offset}"


def parse_offset(text: str) -> timezone:
    """A fixed time zone from text such as +11:00 or -03:30."""
    match = _OFFSET.fullmatch(text)
    if match is None:
        raise OffsetError(f"a UTC offset looks like +11:00 or -03:30, not {text!r}")
    size = timedelta(hours=int(match["hours"]), minutes=int(match["minutes"]))
    if int(match["minutes"]) >= 60 or size > _LARGEST_OFFSET:
        raise OffsetError(f"{text!r} is not a real UTC offset")
    return timezone(size if match["sign"] == "+" else -size)


def format_offset(zone: timezone) -> str:
    """Text such as +11:00 for a fixed time zone."""
    offset = zone.utcoffset(None)
    sign = "-" if offset < timedelta(0) else "+"
    minutes = abs(int(offset.total_seconds())) // 60
    return f"{sign}{minutes // 60:02d}:{minutes % 60:02d}"


def most_common_offset(offsets: Iterable[str | None]) -> str | None:
    """The offset most recordings state, ties going to the first in sorted order."""
    counts = Counter(offset for offset in offsets if offset is not None)
    if not counts:
        return None
    return min(counts, key=lambda offset: (-counts[offset], offset))


def choose_time_basis(requested: str | None, recording_offsets: Iterable[str | None]) -> TimeBasis:
    """The requested offset, or else the one most recordings state.

    When no recording states an offset, times stay on the recorders' own clock.
    """
    offset = requested if requested is not None else most_common_offset(recording_offsets)
    if offset is None:
        return TimeBasis(timezone.utc, None)
    zone = parse_offset(offset)
    return TimeBasis(zone, format_offset(zone))


def window_time(started_at: str, recording_offset: str | None, start_s: int, analysis_zone: timezone) -> datetime:
    """When a window starts, in the analysis time zone.

    A recording without a stated offset is taken to use the analysis offset.
    """
    recording_zone = analysis_zone if recording_offset is None else parse_offset(recording_offset)
    start = datetime.fromisoformat(started_at).replace(tzinfo=recording_zone) + timedelta(seconds=start_s)
    return start.astimezone(analysis_zone)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.analytics.test_timebase`
Expected: all eight tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/analytics/__init__.py src/frog_classifier/analytics/timebase.py tests/analytics/__init__.py tests/analytics/test_timebase.py
git commit -m "feat: place windows in one fixed UTC offset for analysis"
```

---

### Task 4: Effort and activity by hour and date

**Files:**
- Create: `src/frog_classifier/analytics/activity.py`
- Create: `tests/analytics/helpers.py`
- Test: `tests/analytics/test_activity.py`

**Interfaces:**
- Consumes: `DetectionStore.open_read_only`, `recording_rows()`, `windows()`, `identity` (Tasks 2 and Phase 5); `TimeBasis`, `choose_time_basis`, `window_time` (Task 3).
- Produces: `TABLE_THRESHOLDS`, `ActivityError(ValueError)`, `PlacedWindow(recording_path, window_start_s, date, hour, probability)`, `ActivityBin(recorded_windows, recorded_minutes, recordings, detected_windows)` with property `calling_share: float | None`, `ThresholdRow(threshold, detected_windows, calling_share)`, `Coverage(recordings, failed_recordings, recordings_without_time, recordings_without_offset, stated_offsets, unplaced_windows)`, `Activity(identity, record_created_at, threshold, time_basis, windows, total, by_hour, by_date, thresholds, coverage)` with method `is_detected(window) -> bool`, and `read_activity(store, *, threshold=None, utc_offset=None) -> Activity`. `by_hour` is a tuple of 24 bins; `by_date` is a dict holding every date from the first to the last placed window.
- Produces (tests): `tests.analytics.helpers.recording(path, started_at, probabilities, *, utc_offset="+11:00", **changes)`, `build_record(record_dir, recordings, *, threshold=0.5) -> Path`, `read_test_activity(record_dir, recordings, *, model_threshold=0.5, **options) -> Activity`.

- [ ] **Step 1: Write the failing tests**

Create `tests/analytics/helpers.py`:

```python
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

from frog_classifier.analytics.activity import Activity, read_activity
from frog_classifier.inference.store import DATABASE_FILE, DetectionStore, RecordingRow
from tests.inference.helpers import SCAN_IDENTITY, recording_row


def recording(
    path: str,
    started_at: str | None,
    probabilities: Sequence[float],
    *,
    utc_offset: str | None = "+11:00",
    **changes: object,
) -> tuple[RecordingRow, tuple[float, ...]]:
    """A stored recording with one five-second window per probability."""
    row = recording_row(
        path,
        started_at=started_at,
        utc_offset=utc_offset,
        windows=len(probabilities),
        duration_s=5.0 * len(probabilities),
        **changes,
    )
    return row, tuple(probabilities)


def build_record(
    record_dir: Path,
    recordings: Sequence[tuple[RecordingRow, tuple[float, ...]]],
    *,
    threshold: float = 0.5,
) -> Path:
    record_dir.mkdir(parents=True, exist_ok=True)
    identity = replace(SCAN_IDENTITY, threshold=threshold)
    with DetectionStore.open(record_dir / DATABASE_FILE, identity, now="2026-10-15T00:00:00Z") as store:
        for row, probabilities in recordings:
            store.save(row, tuple(range(0, 5 * len(probabilities), 5)), probabilities)
    return record_dir


def read_test_activity(
    record_dir: Path,
    recordings: Sequence[tuple[RecordingRow, tuple[float, ...]]],
    *,
    model_threshold: float = 0.5,
    **options: object,
) -> Activity:
    """Build a record with the given model threshold and read its activity with options."""
    build_record(record_dir, recordings, threshold=model_threshold)
    with DetectionStore.open_read_only(record_dir / DATABASE_FILE) as store:
        return read_activity(store, **options)
```

Create `tests/analytics/test_activity.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from frog_classifier.analytics.activity import ActivityBin, ActivityError, Coverage
from tests.analytics.helpers import read_test_activity, recording


class ReadActivityTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.record = Path(directory.name) / "record"

    def test_effort_and_detections_are_counted_by_hour_of_day(self) -> None:
        activity = read_test_activity(self.record, [
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.9, 0.1, 0.6, 0.2)),
            recording("a/20220124_010000.WAV", "2022-01-24T01:00:00", (0.1, 0.1)),
        ])

        self.assertEqual(activity.by_hour[1], ActivityBin(recorded_windows=6, recorded_minutes=0.5, recordings=2, detected_windows=2))
        self.assertAlmostEqual(activity.by_hour[1].calling_share, 1 / 3)
        self.assertEqual(activity.by_hour[0], ActivityBin(recorded_windows=0, recorded_minutes=0.0, recordings=0, detected_windows=0))
        self.assertIsNone(activity.by_hour[0].calling_share)
        self.assertEqual(activity.total.detected_windows, 2)

    def test_every_date_between_the_first_and_last_is_present(self) -> None:
        activity = read_test_activity(self.record, [
            recording("a/20220120_010000.WAV", "2022-01-20T01:00:00", (0.9,)),
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.9, 0.1)),
        ])

        self.assertEqual(list(activity.by_date), [date(2022, 1, day) for day in (20, 21, 22, 23)])
        self.assertIsNone(activity.by_date[date(2022, 1, 21)].calling_share)
        self.assertEqual(activity.by_date[date(2022, 1, 23)].recorded_windows, 2)

    def test_windows_are_placed_in_one_fixed_offset(self) -> None:
        # The last recorder states standard time; its 00:00 is 01:00 at UTC+11:00.
        recordings = [
            recording("a/20220301_010000.WAV", "2022-03-01T01:00:00", (0.9,)),
            recording("a/20220302_010000.WAV", "2022-03-02T01:00:00", (0.9,)),
            recording("a/20220404_000000.WAV", "2022-04-04T00:00:00", (0.9,), utc_offset="+10:00"),
        ]

        default = read_test_activity(self.record / "default", recordings)
        standard = read_test_activity(self.record / "standard", recordings, utc_offset="+10:00")

        self.assertEqual((default.time_basis.label, default.by_hour[1].recorded_windows), ("UTC+11:00", 3))
        self.assertEqual((standard.time_basis.label, standard.by_hour[0].recorded_windows), ("UTC+10:00", 3))
        self.assertEqual(default.coverage.stated_offsets, ("+10:00", "+11:00"))

    def test_a_recording_that_crosses_midnight_counts_on_both_dates(self) -> None:
        activity = read_test_activity(self.record, [
            recording("a/20220123_235955.WAV", "2022-01-23T23:59:55", (0.9, 0.9)),
        ])

        self.assertEqual(activity.by_date[date(2022, 1, 23)].recorded_windows, 1)
        self.assertEqual(activity.by_date[date(2022, 1, 24)].recorded_windows, 1)
        self.assertEqual((activity.by_hour[23].recorded_windows, activity.by_hour[0].recorded_windows), (1, 1))

    def test_recordings_that_cannot_be_placed_and_failed_recordings_are_counted(self) -> None:
        activity = read_test_activity(self.record, [
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.9, 0.1)),
            recording("a/unnamed.wav", None, (0.9, 0.9), utc_offset=None),
            recording("b/20220123_010000.WAV", "2022-01-23T01:00:00", (0.1, 0.1), utc_offset=None),
            recording("a/broken.wav", None, (), utc_offset=None, status="failed", message="LibsndfileError: bad"),
        ])

        self.assertEqual(activity.coverage, Coverage(
            recordings=3,
            failed_recordings=1,
            recordings_without_time=1,
            recordings_without_offset=1,
            stated_offsets=("+11:00",),
            unplaced_windows=2,
        ))
        self.assertEqual(len(activity.windows), 4)
        self.assertEqual(activity.by_hour[1].recorded_windows, 4)

    def test_a_threshold_override_changes_detections_but_not_effort(self) -> None:
        recordings = [recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.55, 0.95, 0.2))]

        model = read_test_activity(self.record / "model", recordings)
        strict = read_test_activity(self.record / "strict", recordings, threshold=0.9)

        self.assertEqual((model.threshold, model.total.detected_windows), (0.5, 2))
        self.assertEqual((strict.threshold, strict.total.detected_windows), (0.9, 1))
        self.assertEqual(strict.total.recorded_windows, 3)
        self.assertEqual(
            [(row.threshold, row.detected_windows) for row in strict.thresholds],
            [(0.5, 2), (0.7, 1), (0.9, 1), (0.95, 1), (0.99, 0)],
        )

    def test_a_threshold_outside_zero_to_one_is_refused(self) -> None:
        with self.assertRaisesRegex(ActivityError, "between 0 and 1"):
            read_test_activity(self.record, [], threshold=1.5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.analytics.test_activity`
Expected: ERROR with `ModuleNotFoundError: No module named 'frog_classifier.analytics.activity'`.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/analytics/activity.py`:

```python
"""Recording effort and detected activity by hour of day and by date.

The unit is the window. Effort is how many windows were recorded, activity is how
many of them hold a detection, and the calling share is their ratio, which is
undefined, not zero, where nothing was recorded.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

from frog_classifier.inference.store import DetectionStore, ScanIdentity

from .timebase import TimeBasis, choose_time_basis, window_time


# The threshold table always shows these beside the analysis threshold.
TABLE_THRESHOLDS = (0.5, 0.7, 0.9, 0.95, 0.99)


class ActivityError(ValueError):
    pass


@dataclass(frozen=True)
class PlacedWindow:
    recording_path: str
    window_start_s: int
    date: date
    hour: int
    probability: float


@dataclass(frozen=True)
class ActivityBin:
    recorded_windows: int
    recorded_minutes: float
    recordings: int
    detected_windows: int

    @property
    def calling_share(self) -> float | None:
        """Detected windows over recorded windows; None where nothing was recorded."""
        if self.recorded_windows == 0:
            return None
        return self.detected_windows / self.recorded_windows


@dataclass(frozen=True)
class ThresholdRow:
    threshold: float
    detected_windows: int
    calling_share: float | None


@dataclass(frozen=True)
class Coverage:
    recordings: int
    failed_recordings: int
    recordings_without_time: int
    recordings_without_offset: int
    stated_offsets: tuple[str, ...]
    unplaced_windows: int


@dataclass(frozen=True)
class Activity:
    identity: ScanIdentity
    record_created_at: str
    threshold: float
    time_basis: TimeBasis
    windows: tuple[PlacedWindow, ...]
    total: ActivityBin
    by_hour: tuple[ActivityBin, ...]
    by_date: dict[date, ActivityBin]
    thresholds: tuple[ThresholdRow, ...]
    coverage: Coverage

    def is_detected(self, window: PlacedWindow) -> bool:
        return window.probability >= self.threshold


def read_activity(
    store: DetectionStore,
    *,
    threshold: float | None = None,
    utc_offset: str | None = None,
) -> Activity:
    """Place every window of a detection record in time and aggregate it.

    The threshold defaults to the record's model threshold, and the time basis to
    the UTC offset most recordings state.
    """
    identity = store.identity
    threshold = identity.threshold if threshold is None else threshold
    if not 0.0 <= threshold <= 1.0:
        raise ActivityError(f"the threshold must be between 0 and 1, not {threshold}")
    rows = store.recording_rows()
    succeeded = [row for row in rows if row.status == "ok"]
    timed = {row.path: row for row in succeeded if row.started_at is not None}
    basis = choose_time_basis(utc_offset, (row.utc_offset for row in timed.values()))

    windows: list[PlacedWindow] = []
    unplaced = 0
    for path, start_s, probability in store.windows():
        row = timed.get(path)
        if row is None:
            unplaced += 1
            continue
        moment = window_time(row.started_at, row.utc_offset, start_s, basis.zone)
        windows.append(PlacedWindow(path, start_s, moment.date(), moment.hour, probability))

    def summarize(selected: Sequence[PlacedWindow]) -> ActivityBin:
        return ActivityBin(
            recorded_windows=len(selected),
            recorded_minutes=len(selected) * identity.chunk_seconds / 60,
            recordings=len({window.recording_path for window in selected}),
            detected_windows=sum(window.probability >= threshold for window in selected),
        )

    by_hour_windows: dict[int, list[PlacedWindow]] = defaultdict(list)
    by_date_windows: dict[date, list[PlacedWindow]] = defaultdict(list)
    for window in windows:
        by_hour_windows[window.hour].append(window)
        by_date_windows[window.date].append(window)
    by_date: dict[date, ActivityBin] = {}
    if by_date_windows:
        day, last = min(by_date_windows), max(by_date_windows)
        while day <= last:
            by_date[day] = summarize(by_date_windows.get(day, []))
            day += timedelta(days=1)

    return Activity(
        identity=identity,
        record_created_at=store.created_at,
        threshold=threshold,
        time_basis=basis,
        windows=tuple(windows),
        total=summarize(windows),
        by_hour=tuple(summarize(by_hour_windows.get(hour, [])) for hour in range(24)),
        by_date=by_date,
        thresholds=tuple(_threshold_row(windows, value) for value in sorted({*TABLE_THRESHOLDS, threshold})),
        coverage=Coverage(
            recordings=len(succeeded),
            failed_recordings=len(rows) - len(succeeded),
            recordings_without_time=len(succeeded) - len(timed),
            recordings_without_offset=sum(row.utc_offset is None for row in timed.values()),
            stated_offsets=tuple(sorted({row.utc_offset for row in timed.values() if row.utc_offset is not None})),
            unplaced_windows=unplaced,
        ),
    )


def _threshold_row(windows: Sequence[PlacedWindow], threshold: float) -> ThresholdRow:
    detected = sum(window.probability >= threshold for window in windows)
    return ThresholdRow(threshold, detected, detected / len(windows) if windows else None)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.analytics.test_activity tests.analytics.test_timebase`
Expected: every test passes.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/analytics/activity.py tests/analytics/helpers.py tests/analytics/test_activity.py
git commit -m "feat: aggregate recording effort and detections by hour and date"
```

---

### Task 5: Activity figures

**Files:**
- Create: `src/frog_classifier/analytics/figures.py`
- Test: `tests/analytics/test_figures.py`

**Interfaces:**
- Consumes: `Activity`, `ActivityBin` (Task 4).
- Produces: `TIME_OF_DAY_FIGURE = "time_of_day.png"`, `CLOCK_FIGURE = "clock.png"`, `BY_DATE_FIGURE = "by_date.png"`, and `write_time_of_day_figure(path, activity)`, `write_clock_figure(path, activity)`, `write_by_date_figure(path, activity)`, each writing PNG data to `path` whatever its suffix.

- [ ] **Step 1: Write the failing tests**

Create `tests/analytics/test_figures.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from frog_classifier.analytics.figures import (
    write_by_date_figure,
    write_clock_figure,
    write_time_of_day_figure,
)
from tests.analytics.helpers import read_test_activity, recording


WRITERS = (write_time_of_day_figure, write_clock_figure, write_by_date_figure)


class FigureTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _assert_png(self, path: Path) -> None:
        with Image.open(path) as image:
            self.assertEqual(image.format, "PNG")
            self.assertGreater(image.width, 300)

    def test_every_figure_is_a_png_image_even_under_a_temporary_name(self) -> None:
        activity = read_test_activity(self.root / "record", [
            recording("a/20220120_010000.WAV", "2022-01-20T01:00:00", (0.9, 0.1)),
            recording("a/20220123_230000.WAV", "2022-01-23T23:00:00", (0.9, 0.9, 0.2)),
        ])

        for writer in WRITERS:
            with self.subTest(figure=writer.__name__):
                path = self.root / f".{writer.__name__}.png.partial"
                writer(path, activity)
                self._assert_png(path)

    def test_figures_draw_when_nothing_could_be_placed_in_time(self) -> None:
        activity = read_test_activity(self.root / "record", [
            recording("a/unnamed.wav", None, (0.9,), utc_offset=None),
        ])

        for writer in WRITERS:
            with self.subTest(figure=writer.__name__):
                path = self.root / f"{writer.__name__}.png"
                writer(path, activity)
                self._assert_png(path)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.analytics.test_figures`
Expected: ERROR with `ModuleNotFoundError: No module named 'frog_classifier.analytics.figures'`.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/analytics/figures.py`:

```python
"""The activity figures: calling share by hour of day, on a 24-hour clock, and by date.

Hours and dates without recordings are drawn as hatched grey "not recorded"
areas, so missing effort can never look like an absence of calls.
"""

from __future__ import annotations

import math
from datetime import datetime, time, timedelta
from pathlib import Path

from matplotlib import dates as mdates
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from .activity import Activity


TIME_OF_DAY_FIGURE = "time_of_day.png"
CLOCK_FIGURE = "clock.png"
BY_DATE_FIGURE = "by_date.png"

_CALLS = "tab:green"
_EFFORT = "tab:gray"
_UNRECORDED = {"facecolor": "0.93", "edgecolor": "0.75", "hatch": "//", "linewidth": 0}
_DPI = 120


def write_time_of_day_figure(path: Path, activity: Activity) -> None:
    figure = Figure(figsize=(9, 5.5), layout="constrained")
    share_axes, effort_axes = figure.subplots(2, 1, sharex=True, height_ratios=(3, 1))
    recorded = [hour for hour, bin_ in enumerate(activity.by_hour) if bin_.recorded_windows]
    for hour in range(24):
        if hour not in recorded:
            share_axes.axvspan(hour - 0.5, hour + 0.5, **_UNRECORDED)
    shares = [100 * activity.by_hour[hour].calling_share for hour in recorded]
    share_axes.bar(recorded, shares, width=0.8, color=_CALLS)
    share_axes.set(ylim=(0, _share_limit(shares)), ylabel="Calls in recorded time")
    share_axes.yaxis.set_major_formatter(PercentFormatter(100))
    share_axes.legend(handles=_legend(), loc="upper right")
    share_axes.grid(axis="y", alpha=0.3)
    effort_axes.bar(range(24), [bin_.recorded_minutes / 60 for bin_ in activity.by_hour], width=0.8, color=_EFFORT)
    effort_axes.set(ylabel="Recorded hours", xlabel=f"Hour of day ({activity.time_basis.label})", xlim=(-0.5, 23.5))
    effort_axes.set_xticks(range(24), labels=[f"{hour:02d}" for hour in range(24)])
    figure.suptitle(f"Calling activity by hour of day, threshold {activity.threshold:.3f}")
    figure.savefig(path, dpi=_DPI, format="png")


def write_clock_figure(path: Path, activity: Activity) -> None:
    figure = Figure(figsize=(6.5, 6.5), layout="constrained")
    axes = figure.add_subplot(projection="polar")
    axes.set_theta_zero_location("N")
    axes.set_theta_direction(-1)
    width = 2 * math.pi / 24
    shares = [100 * bin_.calling_share for bin_ in activity.by_hour if bin_.recorded_windows]
    limit = _share_limit(shares)
    for hour, bin_ in enumerate(activity.by_hour):
        angle = (hour + 0.5) * width
        if bin_.recorded_windows:
            axes.bar(angle, 100 * bin_.calling_share, width=width * 0.9, color=_CALLS)
        else:
            axes.bar(angle, limit, width=width, **_UNRECORDED)
    axes.set_ylim(0, limit)
    axes.yaxis.set_major_formatter(PercentFormatter(100))
    axes.set_xticks([hour * width for hour in range(0, 24, 3)], labels=[f"{hour:02d}:00" for hour in range(0, 24, 3)])
    axes.set_title(f"Calls in recorded time by hour ({activity.time_basis.label})")
    axes.legend(handles=_legend(), loc="lower center", bbox_to_anchor=(0.5, -0.12), ncols=2)
    figure.savefig(path, dpi=_DPI, format="png")


def write_by_date_figure(path: Path, activity: Activity) -> None:
    figure = Figure(figsize=(10, 5.5), layout="constrained")
    share_axes, effort_axes = figure.subplots(2, 1, sharex=True, height_ratios=(3, 1))
    days = list(activity.by_date)
    recorded = [day for day in days if activity.by_date[day].recorded_windows]
    for day in days:
        if day not in recorded:
            middle = datetime.combine(day, time())
            share_axes.axvspan(middle - timedelta(hours=12), middle + timedelta(hours=12), **_UNRECORDED)
    shares = [100 * activity.by_date[day].calling_share for day in recorded]
    share_axes.bar(recorded, shares, width=0.8, color=_CALLS)
    share_axes.set(ylim=(0, _share_limit(shares)), ylabel="Calls in recorded time")
    share_axes.yaxis.set_major_formatter(PercentFormatter(100))
    share_axes.legend(handles=_legend(), loc="upper right")
    share_axes.grid(axis="y", alpha=0.3)
    effort_axes.bar(days, [activity.by_date[day].recorded_minutes / 60 for day in days], width=0.8, color=_EFFORT)
    effort_axes.set(ylabel="Recorded hours", xlabel=f"Date ({activity.time_basis.label})")
    locator = mdates.AutoDateLocator()
    effort_axes.xaxis.set_major_locator(locator)
    effort_axes.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    if not days:
        share_axes.text(0.5, 0.5, "No recording could be placed in time", transform=share_axes.transAxes, ha="center")
    figure.suptitle(f"Calling activity by date, threshold {activity.threshold:.3f}")
    figure.savefig(path, dpi=_DPI, format="png")


def _share_limit(shares: list[float]) -> float:
    """Room above the tallest bar, and a visible scale even when every share is tiny."""
    return max(5.0, 1.15 * max(shares, default=0.0))


def _legend() -> list[Patch]:
    return [
        Patch(color=_CALLS, label="Calls in recorded time"),
        Patch(label="Not recorded", **_UNRECORDED),
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.analytics.test_figures`
Expected: both tests pass with no warnings printed.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/analytics/figures.py tests/analytics/test_figures.py
git commit -m "feat: draw activity by hour, on a clock, and by date"
```

---

### Task 6: The activity report

**Files:**
- Create: `src/frog_classifier/analytics/report.py`
- Test: `tests/analytics/test_report.py`

**Interfaces:**
- Consumes: `Activity`, `ActivityBin` (Task 4); figure file names (Task 5).
- Produces: `summary_lines(activity: Activity) -> list[str]` and `render_report(activity: Activity, record_name: str) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/analytics/test_report.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from frog_classifier.analytics.report import render_report, summary_lines
from tests.analytics.helpers import read_test_activity, recording
from tests.inference.helpers import SCAN_IDENTITY


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_the_report_states_coverage_unrecorded_time_thresholds_and_provenance(self) -> None:
        activity = read_test_activity(self.root / "record", [
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.9, 0.1)),
            recording("a/20220125_020000.WAV", "2022-01-25T02:00:00", (0.6, 0.6)),
        ])

        report = render_report(activity, "site-a")

        for expected in (
            "# Calling activity: site-a",
            "- Recorded 0.0 hours in 2 recordings on 2 of the 3 dates from 2022-01-23 to 2022-01-25.",
            "- Hours of day recorded: 01:00 to 02:59. Not recorded: 00:00 to 00:59, 03:00 to 23:59, so activity then is unknown.",
            "- Calls were detected in 75.0% of recorded time (3 of 4 windows of 5 seconds) at threshold 0.500.",
            "- Highest calling share: 02:00 to 02:59, 100.0% of recorded time.",
            "- Times are in UTC+11:00 for every recording, so daylight-saving changes cannot shift them.",
            "![Calling share by hour of day](time_of_day.png)",
            "| 00:00 | 0.00 | 0 | 0 | not recorded |",
            "| 01:00 | 0.00 | 1 | 1 | 50.0% |",
            "| 2022-01-24 | 0.00 | 0 | 0 | not recorded |",
            "| 0.990 | 0 | 0.0% |",
            f"| Model run | {SCAN_IDENTITY.model_run} |",
            "| Recordings without a start time, left out | 0 |",
        ):
            with self.subTest(line=expected):
                self.assertIn(expected, report)

    def test_times_without_any_stated_offset_are_the_recorder_clock(self) -> None:
        activity = read_test_activity(self.root / "record", [
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.9,), utc_offset=None),
        ])

        self.assertIn(
            "Times are the recorders' own clock, because no recording states its UTC offset.",
            summary_lines(activity),
        )

    def test_a_record_with_nothing_placed_in_time_says_so(self) -> None:
        activity = read_test_activity(self.root / "record", [
            recording("a/unnamed.wav", None, (0.9,), utc_offset=None),
        ])

        self.assertEqual(
            summary_lines(activity),
            ["No recording could be placed in time, so there is no activity to report."],
        )
        self.assertIn("| Windows that could not be placed in time | 1 |", render_report(activity, "site-a"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.analytics.test_report`
Expected: ERROR with `ModuleNotFoundError: No module named 'frog_classifier.analytics.report'`.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/analytics/report.py`:

```python
"""The analysis report in Markdown, and the summary lines the command prints."""

from __future__ import annotations

from .activity import Activity, ActivityBin
from .figures import BY_DATE_FIGURE, CLOCK_FIGURE, TIME_OF_DAY_FIGURE


FIGURES = (
    (TIME_OF_DAY_FIGURE, "Calling share by hour of day"),
    (CLOCK_FIGURE, "Calling share on a 24-hour clock"),
    (BY_DATE_FIGURE, "Calling share by date"),
)
_TABLE_HEADER = (
    "| {key} | Recorded hours | Recordings | Detected windows | Calls in recorded time |",
    "| --- | ---: | ---: | ---: | ---: |",
)


def summary_lines(activity: Activity) -> list[str]:
    """The headline facts, shared by the report and the command's output."""
    recorded_dates = [day for day, bin_ in activity.by_date.items() if bin_.recorded_windows]
    if not recorded_dates:
        return ["No recording could be placed in time, so there is no activity to report."]
    total = activity.total
    lines = [
        f"Recorded {total.recorded_minutes / 60:.1f} hours in {activity.coverage.recordings} recordings "
        f"on {len(recorded_dates)} of the {len(activity.by_date)} dates from {recorded_dates[0]} to {recorded_dates[-1]}.",
    ]
    hours = f"Hours of day recorded: {_hour_ranges(activity, recorded=True)}."
    unrecorded = _hour_ranges(activity, recorded=False)
    if unrecorded:
        hours += f" Not recorded: {unrecorded}, so activity then is unknown."
    lines.append(hours)
    lines.append(
        f"Calls were detected in {total.calling_share:.1%} of recorded time "
        f"({total.detected_windows:,} of {total.recorded_windows:,} windows of {activity.identity.chunk_seconds} seconds) "
        f"at threshold {activity.threshold:.3f}."
    )
    # max keeps the earliest hour when shares tie.
    peak = max(range(24), key=lambda hour: _share_or_minus_one(activity.by_hour[hour]))
    lines.append(
        f"Highest calling share: {peak:02d}:00 to {peak:02d}:59, "
        f"{activity.by_hour[peak].calling_share:.1%} of recorded time."
    )
    if activity.time_basis.offset is None:
        lines.append("Times are the recorders' own clock, because no recording states its UTC offset.")
    else:
        lines.append(f"Times are in {activity.time_basis.label} for every recording, so daylight-saving changes cannot shift them.")
    return lines


def render_report(activity: Activity, record_name: str) -> str:
    identity, coverage = activity.identity, activity.coverage
    lines = [f"# Calling activity: {record_name}", "", "## Summary", ""]
    lines += [f"- {line}" for line in summary_lines(activity)]
    lines += ["", "## Figures", ""]
    for file_name, title in FIGURES:
        lines += [f"![{title}]({file_name})", ""]
    lines += ["## By hour of day", "", _TABLE_HEADER[0].format(key="Hour"), _TABLE_HEADER[1]]
    lines += [_bin_row(f"{hour:02d}:00", bin_) for hour, bin_ in enumerate(activity.by_hour)]
    lines += ["", "## By date", "", _TABLE_HEADER[0].format(key="Date"), _TABLE_HEADER[1]]
    lines += [_bin_row(day.isoformat(), bin_) for day, bin_ in activity.by_date.items()]
    lines += [
        "",
        "## Threshold",
        "",
        f"A window counts as a detection when its probability is at or above {activity.threshold:.3f}; "
        f"the model's own threshold is {identity.threshold:.3f}.",
        "The same totals at other thresholds show how much the result depends on that choice.",
        "",
        "| Threshold | Detected windows | Calls in recorded time |",
        "| ---: | ---: | ---: |",
    ]
    lines += [
        f"| {row.threshold:.3f} | {row.detected_windows} | {_share(row.calling_share)} |"
        for row in activity.thresholds
    ]
    lines += [
        "",
        "## Provenance",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Detection record | {record_name} |",
        f"| Record created | {activity.record_created_at} |",
        f"| Input folder | {identity.input_root} |",
        f"| Model run | {identity.model_run} |",
        f"| Model SHA-256 | `{identity.model_sha256}` |",
        f"| Preprocessing SHA-256 | `{identity.preprocessing_sha256}` |",
        f"| Model threshold | {identity.threshold:.8f} |",
        f"| Analysis threshold | {activity.threshold:.8f} |",
        f"| Time basis | {activity.time_basis.label} |",
        f"| UTC offsets the recorders state | {', '.join(coverage.stated_offsets) or 'none'} |",
        f"| Recordings analyzed | {coverage.recordings} |",
        f"| Failed recordings | {coverage.failed_recordings} |",
        f"| Recordings without a start time, left out | {coverage.recordings_without_time} |",
        f"| Recordings without a UTC offset, taken to use the time basis | {coverage.recordings_without_offset} |",
        f"| Windows that could not be placed in time | {coverage.unplaced_windows} |",
        "",
    ]
    return "\n".join(lines)


def _bin_row(label: str, bin_: ActivityBin) -> str:
    return (
        f"| {label} | {bin_.recorded_minutes / 60:.2f} | {bin_.recordings} | "
        f"{bin_.detected_windows} | {_share(bin_.calling_share)} |"
    )


def _share(share: float | None) -> str:
    return "not recorded" if share is None else f"{share:.1%}"


def _share_or_minus_one(bin_: ActivityBin) -> float:
    return -1.0 if bin_.calling_share is None else bin_.calling_share


def _hour_ranges(activity: Activity, *, recorded: bool) -> str:
    """Runs of consecutive hours that were, or were not, recorded, such as 00:00 to 11:59."""
    runs: list[list[int]] = []
    for hour, bin_ in enumerate(activity.by_hour):
        if (bin_.recorded_windows > 0) != recorded:
            continue
        if runs and runs[-1][1] == hour - 1:
            runs[-1][1] = hour
        else:
            runs.append([hour, hour])
    return ", ".join(f"{start:02d}:00 to {end:02d}:59" for start, end in runs)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.analytics.test_report`
Expected: all three tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/analytics/report.py tests/analytics/test_report.py
git commit -m "feat: report activity with coverage, thresholds, and provenance"
```

---

### Task 7: One analysis from record to files

**Files:**
- Create: `src/frog_classifier/analytics/analyze.py`
- Test: `tests/analytics/test_analyze.py`

**Interfaces:**
- Consumes: `DetectionStore.open_read_only` (Task 2), `read_activity` (Task 4), the figure writers and names (Task 5), `render_report` (Task 6), `replace_file` (Task 1), `write_csv` from `frog_classifier.training.artifact`.
- Produces: `REPORT_FILE`, `HOUR_TABLE`, `DATE_TABLE`, `WINDOWS_TABLE`, `OUTPUT_FILES`, `default_analysis_dir(results_root: Path, record_dir: Path) -> Path`, `analyze_record(record_dir: Path, output_dir: Path, *, threshold: float | None = None, utc_offset: str | None = None) -> Activity`.

- [ ] **Step 1: Write the failing tests**

Create `tests/analytics/test_analyze.py`:

```python
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from frog_classifier.analytics.analyze import (
    DATE_TABLE,
    HOUR_TABLE,
    OUTPUT_FILES,
    WINDOWS_TABLE,
    analyze_record,
    default_analysis_dir,
)
from frog_classifier.inference.store import DATABASE_FILE, StoreError
from frog_classifier.training.artifact import read_csv
from tests.analytics.helpers import build_record, recording


class AnalyzeRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.record = build_record(self.root / "record", [
            recording("a/20220120_010000.WAV", "2022-01-20T01:00:00", (0.9, 0.1, 0.6)),
            recording("a/20220122_235955.WAV", "2022-01-22T23:59:55", (0.95, 0.2, 0.7)),
            recording("a/20220123_020000.WAV", "2022-01-23T02:00:00", (0.1, 0.1)),
        ])
        self.output = self.root / "analysis"

    def test_every_output_is_written(self) -> None:
        analyze_record(self.record, self.output)

        self.assertEqual(sorted(path.name for path in self.output.iterdir()), sorted(OUTPUT_FILES))

    def test_every_hour_and_date_value_is_reproduced_from_the_windows_table(self) -> None:
        analyze_record(self.record, self.output)

        windows = read_csv(self.output / WINDOWS_TABLE)
        for table, key in ((HOUR_TABLE, "hour"), (DATE_TABLE, "date")):
            for row in read_csv(self.output / table):
                with self.subTest(table=table, bin=row[key]):
                    matching = [window for window in windows if window[key] == row[key]]
                    detected = sum(int(window["detected"]) for window in matching)
                    self.assertEqual(int(row["recorded_windows"]), len(matching))
                    self.assertEqual(int(row["detected_windows"]), detected)
                    self.assertEqual(int(row["recordings"]), len({window["recording_path"] for window in matching}))
                    expected_share = f"{detected / len(matching):.6f}" if matching else ""
                    self.assertEqual(row["calling_share"], expected_share)

    def test_the_detection_record_is_left_unchanged(self) -> None:
        before = {path.name: path.read_bytes() for path in self.record.iterdir()}

        analyze_record(self.record, self.output)

        self.assertEqual({path.name: path.read_bytes() for path in self.record.iterdir()}, before)

    def test_a_folder_without_a_record_is_refused(self) -> None:
        with self.assertRaisesRegex(StoreError, "run the detection command first"):
            analyze_record(self.root / "empty", self.output)

    def test_the_default_output_folder_is_named_after_the_record(self) -> None:
        self.assertEqual(default_analysis_dir(self.root / "results", self.record), self.root / "results" / "record")

    def test_analysis_does_not_load_pytorch(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import sys, frog_classifier.analytics.analyze; print('torch' in sys.modules)"],
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.analytics.test_analyze`
Expected: ERROR with `ModuleNotFoundError: No module named 'frog_classifier.analytics.analyze'`.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/analytics/analyze.py`:

```python
"""One analysis: read a detection record, aggregate it, and write every output."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from frog_classifier.files import replace_file
from frog_classifier.inference.store import DATABASE_FILE, DetectionStore
from frog_classifier.training.artifact import write_csv

from .activity import Activity, ActivityBin, read_activity
from .figures import (
    BY_DATE_FIGURE,
    CLOCK_FIGURE,
    TIME_OF_DAY_FIGURE,
    write_by_date_figure,
    write_clock_figure,
    write_time_of_day_figure,
)
from .report import render_report


REPORT_FILE = "report.md"
HOUR_TABLE = "activity_by_hour.csv"
DATE_TABLE = "activity_by_date.csv"
WINDOWS_TABLE = "windows.csv"
OUTPUT_FILES = (
    REPORT_FILE, TIME_OF_DAY_FIGURE, CLOCK_FIGURE, BY_DATE_FIGURE, HOUR_TABLE, DATE_TABLE, WINDOWS_TABLE,
)
_BIN_COLUMNS = ("recorded_windows", "recorded_minutes", "recordings", "detected_windows", "calling_share")
_WINDOW_COLUMNS = ("recording_path", "window_start_s", "date", "hour", "probability", "detected")


def default_analysis_dir(results_root: Path, record_dir: Path) -> Path:
    return results_root / record_dir.resolve().name


def analyze_record(
    record_dir: Path,
    output_dir: Path,
    *,
    threshold: float | None = None,
    utc_offset: str | None = None,
) -> Activity:
    """Analyze one detection record into output_dir, replacing any earlier analysis there.

    The record is opened read-only and never changes.
    """
    with DetectionStore.open_read_only(record_dir / DATABASE_FILE) as store:
        activity = read_activity(store, threshold=threshold, utc_offset=utc_offset)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_table(output_dir / HOUR_TABLE, ("hour", *_BIN_COLUMNS), (
        {"hour": hour, **_bin_values(bin_)} for hour, bin_ in enumerate(activity.by_hour)
    ))
    _write_table(output_dir / DATE_TABLE, ("date", *_BIN_COLUMNS), (
        {"date": day.isoformat(), **_bin_values(bin_)} for day, bin_ in activity.by_date.items()
    ))
    _write_table(output_dir / WINDOWS_TABLE, _WINDOW_COLUMNS, (
        {
            "recording_path": window.recording_path,
            "window_start_s": window.window_start_s,
            "date": window.date.isoformat(),
            "hour": window.hour,
            "probability": f"{window.probability:.8f}",
            "detected": int(activity.is_detected(window)),
        }
        for window in activity.windows
    ))
    for name, write in (
        (TIME_OF_DAY_FIGURE, write_time_of_day_figure),
        (CLOCK_FIGURE, write_clock_figure),
        (BY_DATE_FIGURE, write_by_date_figure),
    ):
        replace_file(output_dir / name, lambda staging, write=write: write(staging, activity))
    report = render_report(activity, record_dir.resolve().name).encode("utf-8")
    replace_file(output_dir / REPORT_FILE, lambda staging: staging.write_bytes(report))
    return activity


def _bin_values(bin_: ActivityBin) -> dict[str, object]:
    return {
        "recorded_windows": bin_.recorded_windows,
        "recorded_minutes": f"{bin_.recorded_minutes:.2f}",
        "recordings": bin_.recordings,
        "detected_windows": bin_.detected_windows,
        "calling_share": None if bin_.calling_share is None else f"{bin_.calling_share:.6f}",
    }


def _write_table(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    replace_file(path, lambda staging: write_csv(staging, columns, rows))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.analytics.test_analyze`
Expected: all six tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/analytics/analyze.py tests/analytics/test_analyze.py
git commit -m "feat: analyze a detection record into a report, figures, and tables"
```

---

### Task 8: The analysis command, repository checks, and workflow

**Files:**
- Create: `scripts/analyze_activity.py`
- Modify: `tests/test_repository_reproducibility.py`
- Modify: `docs/workflow.md`
- Test: `tests/analytics/test_command.py`

**Interfaces:**
- Consumes: `analyze_record`, `default_analysis_dir`, `REPORT_FILE` (Task 7), `summary_lines` (Task 6), `ActivityError` (Task 4), `OffsetError` (Task 3), `FileInUseError` (Task 1), `StoreError`.
- Produces: `scripts/analyze_activity.py` with `main() -> int`: 0 on success, 1 when a file is open elsewhere, 2 when the analysis cannot start.

- [ ] **Step 1: Write the failing tests**

Create `tests/analytics/test_command.py`:

```python
from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tests.analytics.helpers import build_record, recording


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_activity.py"


def _load(repo_root: Path):
    specification = importlib.util.spec_from_file_location("analyze_activity_under_test", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    # The command finds the repository from its own location.
    module.__file__ = str(repo_root / "scripts" / "analyze_activity.py")
    return module


def _run(module, *arguments: str) -> tuple[int, str, str]:
    output, errors = io.StringIO(), io.StringIO()
    with patch.object(sys, "argv", ["analyze_activity.py", *arguments]), redirect_stdout(output), redirect_stderr(errors):
        code = module.main()
    return code, output.getvalue(), errors.getvalue()


class AnalyzeCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo = Path(directory.name)
        build_record(self.repo / "results" / "detections" / "site-run", [
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.95, 0.1, 0.6)),
        ])
        self.command = _load(self.repo)

    def test_analyzes_into_the_default_folder_and_summarizes(self) -> None:
        code, output, _ = _run(self.command, "results/detections/site-run")

        self.assertEqual(code, 0)
        self.assertIn("Calls were detected in 66.7% of recorded time", output)
        report = self.repo / "results" / "analytics" / "site-run" / "report.md"
        self.assertTrue(report.is_file())
        self.assertIn(f"Report: {report}", output)

    def test_overrides_reach_the_analysis(self) -> None:
        code, output, _ = _run(
            self.command, "results/detections/site-run",
            "--threshold", "0.9", "--utc-offset", "+10:00", "--output", "results/analytics/custom",
        )

        self.assertEqual(code, 0)
        self.assertIn("at threshold 0.900", output)
        report = (self.repo / "results" / "analytics" / "custom" / "report.md").read_bytes().decode("utf-8")
        self.assertIn("| Time basis | UTC+10:00 |", report)

    def test_a_folder_without_a_record_cannot_be_analyzed(self) -> None:
        code, _, errors = _run(self.command, "results/detections/missing")

        self.assertEqual(code, 2)
        self.assertIn("Cannot analyze:", errors)

    def test_a_malformed_offset_cannot_be_analyzed(self) -> None:
        code, _, errors = _run(self.command, "results/detections/site-run", "--utc-offset", "11")

        self.assertEqual(code, 2)
        self.assertIn("a UTC offset looks like +11:00", errors)


if __name__ == "__main__":
    unittest.main()
```

In `tests/test_repository_reproducibility.py`:

- Add after `INFERENCE_PACKAGE_MODULES`:

```python
ANALYTICS_PACKAGE_MODULES = tuple(
    f"src/frog_classifier/analytics/{name}.py"
    for name in ("__init__", "activity", "analyze", "figures", "report", "timebase")
)
```

- Add `"src/frog_classifier/files.py",` and `*ANALYTICS_PACKAGE_MODULES,` after `*INFERENCE_PACKAGE_MODULES,` in `test_project_navigation_files_are_complete_and_linked`.
- Add `"scripts/analyze_activity.py",` to `OPERATIONAL_TEXT_FILES` before `"scripts/build_manifest.py",`.
- Add `"uv run python scripts/analyze_activity.py",` to the command list in `test_workflow_documents_every_operational_command` after the `detect_calls.py` line.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.analytics.test_command tests.test_repository_reproducibility`
Expected: the command tests ERROR with `FileNotFoundError` for `scripts/analyze_activity.py`; the repository tests FAIL for the script and the workflow command.

- [ ] **Step 3: Implement**

Create `scripts/analyze_activity.py`:

```python
"""Summarize calling activity in a detection record by hour of day and by date.

The analysis goes to one folder: a report, three figures, and tables that trace
every figure value back to individual windows. Recording effort is always shown
beside detected calls, so hours that were never recorded cannot look silent.
Running the command again replaces those files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from frog_classifier.analytics.activity import ActivityError
from frog_classifier.analytics.analyze import REPORT_FILE, analyze_record, default_analysis_dir
from frog_classifier.analytics.report import summary_lines
from frog_classifier.analytics.timebase import OffsetError
from frog_classifier.files import FileInUseError
from frog_classifier.inference.store import StoreError


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("record", type=Path, help="Detection record folder, for example results/detections/external-resnet18-20261014T114432Z.")
    parser.add_argument("--threshold", type=float, default=None, help="Detection threshold. Default: the model's own threshold.")
    parser.add_argument(
        "--utc-offset",
        default=None,
        help="One fixed UTC offset for every time, such as +10:00; write a negative one as --utc-offset=-03:00. "
        "Default: the offset the recorders state.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output folder. Default: results/analytics/<record folder>.")
    args = parser.parse_args()

    record_dir = _resolve(repo_root, args.record)
    if args.output is not None:
        output_dir = _resolve(repo_root, args.output)
    else:
        output_dir = default_analysis_dir(repo_root / "results" / "analytics", record_dir)
    try:
        activity = analyze_record(record_dir, output_dir, threshold=args.threshold, utc_offset=args.utc_offset)
    except (StoreError, OffsetError, ActivityError) as error:
        print(f"Cannot analyze: {error}", file=sys.stderr)
        return 2
    except FileInUseError as error:
        print(error, file=sys.stderr)
        return 1
    for line in summary_lines(activity):
        print(line)
    print(f"Report: {output_dir / REPORT_FILE}")
    return 0


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


if __name__ == "__main__":
    raise SystemExit(main())
```

In `docs/workflow.md`, add this section after "Detect calls in recordings" and before "Verify Phase 2 acceptance":

````markdown
## Analyze calling activity

`scripts/analyze_activity.py` turns one detection record into calling activity by hour of day and by date.

```powershell
uv run python scripts/analyze_activity.py results/detections/<record folder>
```

The analysis goes to `results/analytics/<record folder>/` unless `--output` names another folder, and running it again replaces those files.
It opens the detection record read-only and never changes it.

| File | Contents |
| --- | --- |
| `report.md` | Coverage, the peak hour, tables by hour and by date, the threshold table, the figures, and provenance |
| `time_of_day.png` | The share of recorded time with calls for each hour of day, above the hours recorded |
| `clock.png` | The same shares on a 24-hour clock with midnight at the top |
| `by_date.png` | The share of recorded time with calls for each date, above the hours recorded |
| `activity_by_hour.csv`, `activity_by_date.csv` | Recorded windows and minutes, recordings, detected windows, and the calling share for each hour and date |
| `windows.csv` | Every window with its recording, date, hour, probability, and whether it counts as a detection |

The calling share is detected windows divided by recorded windows.
Hours and dates without recordings are marked as not recorded, never as zero calls, because nothing is known about them.
Every value in the tables and figures can be reproduced by filtering `windows.csv`.

Times use one fixed UTC offset for every recording, by default the one the recorders state, so daylight-saving changes cannot shift activity by an hour.
Choose another fixed offset, or another threshold, like this:

```powershell
uv run python scripts/analyze_activity.py results/detections/<record folder> --utc-offset +10:00 --threshold 0.9 --output results/analytics/<name>
```

The report repeats the totals at thresholds from 0.5 to 0.99 so it is clear how much the result depends on the threshold.
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.analytics.test_command tests.test_repository_reproducibility`
Expected: every test passes.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_activity.py tests/analytics/test_command.py tests/test_repository_reproducibility.py docs/workflow.md
git commit -m "feat: add the activity analysis command"
```

---

### Task 9: Real-data analysis and acceptance

**Files:**
- No tracked files change. Output goes to `results/analytics/`.

- [ ] **Step 1: Full verification**

Run the full suite to a log file, `compileall` over `scripts` and `src`, `uv lock --check`, and `git diff --check`.
Expected: the suite reports `OK` and the other checks print nothing unexpected.

- [ ] **Step 2: Analyze the real record**

Run: `.venv/Scripts/python.exe scripts/analyze_activity.py results/detections/external-resnet18-20261014T114432Z`
Expected: 618 recordings on 58 of the 76 dates from 2022-01-20 to 2022-04-05, hours 00:00 to 11:59 recorded and 12:00 to 23:59 not recorded, 8,080 of 37,080 windows detected at threshold 0.794, times in UTC+11:00.

- [ ] **Step 3: Check the numbers**

With a scratch script, reproduce every hour and date row from `windows.csv`, and compare the hour shares with a direct query of the detection record grouped by the hour of `started_at`.
Expected: every value matches.

- [ ] **Step 4: Inspect the figures**

Open the three PNG files and confirm that unrecorded hours and dates are hatched grey, the clock has midnight at the top and runs clockwise, and the labels state UTC+11:00 and the threshold.

---

### Task 10: Documentation and project status

**Files:**
- Modify: `docs/workflow.md`, `docs/architecture.md`, `docs/outline.md`, `README.md`, `roadmap.md`, `AGENTS.md`

- [ ] **Step 1: Update the documents**

- `docs/workflow.md`: add the real-data result to "Analyze calling activity".
- `docs/architecture.md`: add an activity-analytics section with the module table, the time basis, effort and the calling share, traceability, read-only access, and the shared `frog_classifier.files`; update the storage contract line for `results/analytics/` and the pipeline diagram.
- `docs/outline.md` and `README.md`: describe the analysis command and its place in the pipeline.
- `roadmap.md`: record that Phase 6 is in progress with part 1 done and verified, and list the remaining parts.
- `AGENTS.md`: Phase 6 in progress.

- [ ] **Step 2: Verify and commit**

Run the full suite, `git diff --check`, and the em dash scan over every changed file, then commit the documents.

## Execution record

Executed inline on `main` on October 15, 2026, with a final self-review instead of a separate reviewer.

Rulings made during execution:

- The work ran on `main` in the current checkout, as Phases 4 and 5 did.
- The threshold check lives in `read_activity` as `ActivityError` rather than in `analyze.py` as the spec's `AnalysisError`, because the application will call `read_activity` directly and the check belongs where the threshold is used.
- The writer's "choose another output folder" advice moved into one place in `DetectionStore.open`, so the read-only path can give its own advice; two existing refusal tests now pin that advice, and a read-only test covers an SQLite file without a `scan` table.
- The figure tests turn matplotlib warnings into errors, so "no warnings" is asserted rather than eyeballed.
- On the real clock figure the percentage labels covered the tallest bars; they now sit along the quietest hour.
- The roadmap test required the literal `Status: Next`, which no phase has once Phase 6 is honestly in progress; it now requires every phase to carry one status from Complete, In progress, Next, or Planned.
- The final review found that the by-date figure took 41 seconds for a record spanning 1970 to 2022, which one recorder with an unset clock produces, because it drew one hatched span per missing date; each run of missing dates is now one span, with a test, and the figure takes 0.2 seconds.
- Analysis at the same moment as a detection run on the same record is not supported yet: the journal check refuses only while a transaction is open, and the two reads are separate snapshots; the application part must decide whether analysis waits for detection or the record moves to SQLite's WAL mode.
- Implausible recording dates such as 1970 are reported rather than filtered: the summary states the date span and how many dates were recorded, which exposes an unset clock, and the application can turn that into a warning.

Deferred minor items:

- The summary counts every successful recording but the hours count only placed windows, so recordings without a start time inflate "in N recordings".
- A file open in another program stops the analysis partway, leaving a mix of new and old files until the rerun.
- `replace_file` leaves its `.partial` file if the write itself fails, for example on a full disk.
- A crashed detection run whose journal appears between the journal check and the connection gives the "not a detection database" message instead of the interrupted-run advice.
