# Label Audit and Unsure Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give uncertain clips a recorded home, let the existing labels be audited safely, and order new clips night-first, through a small `frog_classifier.labeling` package and a rewritten Streamlit labeler, with the terminal labeler retired.

**Architecture:** Folders stay the source of truth for training data; `labeled/unsure/` is a holding folder the manifest ignores, and `labeled/decisions.csv` is an append-only log of every button press. The package owns the queue, the moves, the log, progress counts, and audio export; the Streamlit script stays thin.

**Tech Stack:** Python 3.13, NumPy, librosa (audio window load), standard library `csv` and `wave`, Streamlit, unittest.

**Spec:** `docs/superpowers/specs/2026-09-13-label-audit-and-unsure-workflow-design.md`

## Global Constraints

- Python 3.13 only; `pyproject.toml` and `uv.lock` do not change. No new dependencies.
- Run tests with `.venv/Scripts/python.exe -m unittest ...` from the repository root (`uv` is not on the PATH here; the documented user command stays `uv run python ...`).
- Every test uses synthetic files in a temporary directory. Never read or write `raw/`, `processed/`, or `labeled/` from a test.
- Never delete, rename, or overwrite anything under `raw/` or `labeled/`. Moves in this plan go only to destinations that do not exist yet.
- LF line endings; use the Write and Edit tools; run `git diff --check` before every commit. No em dash characters. Markdown: one full sentence per physical line.
- Commit messages are a single subject line with no body and no trailer of any kind (no Co-Authored-By).
- Work on branch `feature/label-audit-and-unsure` created from `main`, in the main checkout `D:\projects\frog-classifier`.
- Naming contract: example files are `<recording_id>_start<N>s.png`; `frog_classifier.data.naming.parse_example_filename(path, *, chunk_seconds)` returns `ExampleKey(recording_id, start_s)` with `.example_id`, and raises `ExampleNameError` otherwise. `chunk_seconds` is 5 in the tracked config.
- Decision folders are exactly `litoria_aurea`, `non_target`, and `unsure`. The log file is `labeled/decisions.csv` with columns `example_id,decision,action,faint,decided_at`. Night hours are 19 through 23 and 0 through 6.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/frog_classifier/data/manifest.py` | Skip the `unsure` holding folder during discovery. |
| `labeled/unsure/.gitkeep` | Tracked placeholder for the holding folder. |
| `src/frog_classifier/labeling/__init__.py` | Public exports. |
| `src/frog_classifier/labeling/queue.py` | Night-first session order with a per-recording cap. |
| `src/frog_classifier/labeling/decisions.py` | Decision log, moves, confirm, undo, progress counts. |
| `src/frog_classifier/labeling/audio.py` | Locate or export the playback WAV. |
| `scripts/label_frontend.py` | Streamlit labeler with label and audit modes (rewritten). |
| `scripts/label_spectrograms.py` | Deleted. |
| `tests/labeling/` | Package tests. |
| `tests/data/test_manifest.py`, `tests/test_repository_reproducibility.py` | Updated. |
| `README.md`, `AGENTS.md`, `docs/outline.md`, `docs/architecture.md`, `docs/workflow.md`, `roadmap.md` | Documentation. |

---

### Task 1: Manifest ignores the holding folder

**Files:**
- Modify: `src/frog_classifier/data/manifest.py`
- Create: `labeled/unsure/.gitkeep` (empty)
- Modify: `tests/data/test_manifest.py`
- Modify: `tests/test_repository_reproducibility.py`

**Interfaces:**
- Produces: `HOLDING_DIRECTORIES = ("unsure",)` in `frog_classifier.data.manifest`, exported from `frog_classifier.data`.

- [ ] **Step 1: Write the failing test**

In `tests/data/test_manifest.py`, after `test_reports_unknown_label_directory`, add:

```python
    def test_skips_the_unsure_holding_directory(self) -> None:
        examples = self.discover((
            "litoria_aurea/recording_start5s.png",
            "non_target/other_start10s.png",
            "unsure/third_start15s.png",
        ))

        self.assertEqual(
            [example.example_id for example in examples],
            ["recording_start5s", "other_start10s"],
        )
```

Read the class's existing `discover` helper first and call it the same way the neighbouring tests do; if it returns something other than the tuple of examples, adapt the assertion to what it returns.

In `tests/test_repository_reproducibility.py`, add `"labeled/unsure/.gitkeep",` to `PLACEHOLDERS` after `"labeled/non_target/.gitkeep",`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.data.test_manifest tests.test_repository_reproducibility -v`
Expected: the new manifest test fails with a `ManifestValidationError` carrying `unknown_label_directory`; the placeholder test fails because the file does not exist.

- [ ] **Step 3: Implement**

In `src/frog_classifier/data/manifest.py`, after `MANIFEST_COLUMNS`, add:

```python
# Folders under the label root that hold clips outside training, such as
# clips the reviewer could not decide on. Discovery skips them silently.
HOLDING_DIRECTORIES = ("unsure",)
```

In `discover_labeled_examples`, immediately after `label_name = relative_to_labels.parts[0]`, add:

```python
        if label_name in HOLDING_DIRECTORIES:
            continue
```

Export it: in `src/frog_classifier/data/__init__.py` add `HOLDING_DIRECTORIES,` to the `from .manifest import (...)` list after `LabeledExample,` and `"HOLDING_DIRECTORIES",` to `__all__` after `"ExampleNameError",`.

Create the empty file `labeled/unsure/.gitkeep`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.data.test_manifest tests.test_repository_reproducibility -v`
Expected: OK.

- [ ] **Step 5: Full suite and commit**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: 135 tests, OK.

```bash
git add src/frog_classifier/data/manifest.py src/frog_classifier/data/__init__.py labeled/unsure/.gitkeep tests/data/test_manifest.py tests/test_repository_reproducibility.py
git diff --cached --check
git commit -m "feat: keep unsure clips out of the manifest"
```

---

### Task 2: Night-first queue

**Files:**
- Create: `src/frog_classifier/labeling/__init__.py`
- Create: `src/frog_classifier/labeling/queue.py`
- Create: `tests/labeling/__init__.py` (empty)
- Create: `tests/labeling/test_queue.py`

**Interfaces:**
- Produces: `hour_of_day(recording_id: str) -> int | None`, `is_night(recording_id: str) -> bool`, `build_queue(image_paths, *, seed: int, chunk_seconds: int = 5, per_recording_cap: int = 5, daytime_share: float = 0.1) -> list[Path]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/labeling/test_queue.py`:

```python
from __future__ import annotations

import unittest
from pathlib import Path

from frog_classifier.labeling.queue import build_queue, hour_of_day, is_night


def clip(recording_id: str, start_s: int = 0) -> Path:
    return Path("queue") / f"{recording_id}_start{start_s}s.png"


class HourOfDayTests(unittest.TestCase):
    def test_reads_the_hour_from_a_timestamped_recording_id(self) -> None:
        self.assertEqual(hour_of_day("20220121_030000"), 3)
        self.assertEqual(hour_of_day("2MM03935_20251130_180000"), 18)

    def test_missing_timestamp_is_unknown_and_counts_as_daytime(self) -> None:
        self.assertIsNone(hour_of_day("pond_recording"))
        self.assertFalse(is_night("pond_recording"))

    def test_night_spans_seven_pm_to_seven_am(self) -> None:
        for hour in (19, 23, 0, 6):
            self.assertTrue(is_night(f"20220101_{hour:02d}0000"), hour)
        for hour in (7, 12, 18):
            self.assertFalse(is_night(f"20220101_{hour:02d}0000"), hour)


class BuildQueueTests(unittest.TestCase):
    def test_interleaves_nine_night_clips_with_one_daytime_clip(self) -> None:
        night = [clip(f"202201{day:02d}_220000") for day in range(1, 19)]
        day = [clip(f"202201{day:02d}_100000") for day in range(1, 5)]

        queue = build_queue(night + day, seed=1)

        pattern = [is_night(path.name) for path in queue]
        self.assertEqual(len(queue), 22)
        self.assertEqual(pattern[:10], [True] * 9 + [False])
        self.assertEqual(pattern[10:20], [True] * 9 + [False])
        self.assertEqual(pattern[20:], [False, False])

    def test_caps_clips_per_recording(self) -> None:
        clips = [clip("20220101_220000", start_s=5 * index) for index in range(12)]

        queue = build_queue(clips, seed=1, per_recording_cap=5)

        self.assertEqual(len(queue), 5)

    def test_same_seed_same_order_and_different_seed_different_order(self) -> None:
        clips = [clip(f"202201{day:02d}_220000") for day in range(1, 25)]

        first = build_queue(clips, seed=7)
        second = build_queue(clips, seed=7)
        third = build_queue(clips, seed=8)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_skips_files_that_do_not_follow_the_naming_contract(self) -> None:
        clips = [clip("20220101_220000"), Path("queue") / "notes.png"]

        self.assertEqual(build_queue(clips, seed=1), [clip("20220101_220000")])

    def test_rejects_invalid_cap_and_share(self) -> None:
        with self.assertRaisesRegex(ValueError, "cap"):
            build_queue([], seed=1, per_recording_cap=0)
        with self.assertRaisesRegex(ValueError, "share"):
            build_queue([], seed=1, daytime_share=1.0)


if __name__ == "__main__":
    unittest.main()
```

Create `tests/labeling/__init__.py` as an empty file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.labeling.test_queue -v`
Expected: FAIL at import with `ModuleNotFoundError: No module named 'frog_classifier.labeling'`.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/labeling/queue.py`:

```python
from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from frog_classifier.data.naming import ExampleNameError, parse_example_filename


NIGHT_HOURS = frozenset(range(19, 24)) | frozenset(range(0, 7))

# A recording ID such as 20220121_030000 or 2MM03935_20251130_180000
# carries the start time as YYYYMMDD_HHMMSS.
_TIMESTAMP = re.compile(r"(?<!\d)\d{8}_(\d{2})\d{4}(?!\d)")


def hour_of_day(recording_id: str) -> int | None:
    """Hour from the recording ID's timestamp, or None when it has none."""
    match = _TIMESTAMP.search(recording_id)
    if match is None:
        return None
    hour = int(match.group(1))
    return hour if hour < 24 else None


def is_night(recording_id: str) -> bool:
    hour = hour_of_day(recording_id)
    return hour is not None and hour in NIGHT_HOURS


def build_queue(
    image_paths: Iterable[Path],
    *,
    seed: int,
    chunk_seconds: int = 5,
    per_recording_cap: int = 5,
    daytime_share: float = 0.1,
) -> list[Path]:
    """Session order: shuffled night clips, capped per recording, with a
    share of daytime clips interleaved. Files that do not follow the naming
    contract are left out because they could never be labeled."""
    if per_recording_cap < 1:
        raise ValueError("per_recording_cap must be at least 1")
    if not 0.0 <= daytime_share < 1.0:
        raise ValueError("daytime_share must be at least 0 and below 1")

    night: list[tuple[str, Path]] = []
    day: list[tuple[str, Path]] = []
    for path in sorted(image_paths, key=lambda candidate: candidate.as_posix()):
        try:
            key = parse_example_filename(path, chunk_seconds=chunk_seconds)
        except ExampleNameError:
            continue
        (night if is_night(key.recording_id) else day).append((key.recording_id, path))

    rng = random.Random(seed)
    return _interleave(
        _cap(_shuffled(night, rng), per_recording_cap),
        _cap(_shuffled(day, rng), per_recording_cap),
        daytime_share,
    )


def _shuffled(items: Sequence[tuple[str, Path]], rng: random.Random) -> list[tuple[str, Path]]:
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled


def _cap(items: Sequence[tuple[str, Path]], cap: int) -> list[Path]:
    seen: dict[str, int] = defaultdict(int)
    kept: list[Path] = []
    for recording_id, path in items:
        if seen[recording_id] < cap:
            seen[recording_id] += 1
            kept.append(path)
    return kept


def _interleave(night: list[Path], day: list[Path], daytime_share: float) -> list[Path]:
    if daytime_share == 0.0:
        return night + day
    run = max(1, round((1.0 - daytime_share) / daytime_share))
    result: list[Path] = []
    next_night = 0
    next_day = 0
    while next_night < len(night) or next_day < len(day):
        for _ in range(run):
            if next_night < len(night):
                result.append(night[next_night])
                next_night += 1
        if next_day < len(day):
            result.append(day[next_day])
            next_day += 1
    return result
```

Create `src/frog_classifier/labeling/__init__.py`:

```python
from .queue import NIGHT_HOURS, build_queue, hour_of_day, is_night

__all__ = [
    "NIGHT_HOURS",
    "build_queue",
    "hour_of_day",
    "is_night",
]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.labeling.test_queue -v`
Expected: `Ran 8 tests`, OK.

- [ ] **Step 5: Full suite and commit**

```bash
git add src/frog_classifier/labeling tests/labeling
git diff --cached --check
git commit -m "feat: order labeling sessions night-first with a per-recording cap"
```

---

### Task 3: Decision log, moves, undo, and progress

**Files:**
- Create: `src/frog_classifier/labeling/decisions.py`
- Modify: `src/frog_classifier/labeling/__init__.py`
- Create: `tests/labeling/test_decisions.py`

**Interfaces:**
- Consumes: `HOLDING_DIRECTORIES` from Task 1.
- Produces: `CLASS_FOLDERS`, `HOLDING_FOLDER`, `DECISION_FOLDERS`, `LOG_COLUMNS`, `LOG_NAME`; `DecisionRow`, `Move`, `ClassProgress`; `DecisionLog(path)` with `.append(example_id, decision, action, *, faint=False) -> DecisionRow` and `.rows() -> list[DecisionRow]`; `current_folder(image_path, labeled_root) -> str | None`; `record_decision(image_path, decision, *, labeled_root, log, faint=False, chunk_seconds=5) -> Move`; `confirm_decision(image_path, *, labeled_root, log, chunk_seconds=5) -> DecisionRow`; `undo_move(move, *, labeled_root, log, chunk_seconds=5) -> Path`; `summarize_labels(labeled_root, *, chunk_seconds=5) -> dict[str, ClassProgress]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/labeling/test_decisions.py`:

```python
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from frog_classifier.labeling.decisions import (
    LOG_COLUMNS,
    DecisionLog,
    confirm_decision,
    current_folder,
    record_decision,
    summarize_labels,
    undo_move,
)


class DecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.labeled_root = self.root / "labeled"
        self.queue_root = self.root / "queue"
        self.log = DecisionLog(self.labeled_root / "decisions.csv")
        self.clip = self._write(self.queue_root / "site" / "20220101_220000_start5s.png")

    def test_label_moves_the_clip_and_logs_a_label_row(self) -> None:
        move = record_decision(self.clip, "litoria_aurea", labeled_root=self.labeled_root, log=self.log, faint=True)

        self.assertFalse(self.clip.exists())
        self.assertEqual(move.destination, self.labeled_root / "litoria_aurea" / self.clip.name)
        self.assertTrue(move.destination.is_file())
        rows = self.log.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0].example_id, rows[0].decision, rows[0].action, rows[0].faint), ("20220101_220000_start5s", "litoria_aurea", "label", True))
        self.assertRegex(rows[0].decided_at, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_log_is_created_with_a_header_once(self) -> None:
        self.log.append("a_start0s", "unsure", "label")
        self.log.append("b_start0s", "non_target", "label")

        with self.log.path.open(newline="", encoding="utf-8") as handle:
            lines = list(csv.reader(handle))
        self.assertEqual(lines[0], list(LOG_COLUMNS))
        self.assertEqual(len(lines), 3)

    def test_change_moves_between_labeled_folders_and_logs_change(self) -> None:
        move = record_decision(self.clip, "non_target", labeled_root=self.labeled_root, log=self.log)

        changed = record_decision(move.destination, "unsure", labeled_root=self.labeled_root, log=self.log)

        self.assertEqual(changed.destination, self.labeled_root / "unsure" / self.clip.name)
        self.assertFalse(move.destination.exists())
        self.assertEqual([row.action for row in self.log.rows()], ["label", "change"])

    def test_confirm_logs_without_moving(self) -> None:
        move = record_decision(self.clip, "non_target", labeled_root=self.labeled_root, log=self.log)

        row = confirm_decision(move.destination, labeled_root=self.labeled_root, log=self.log)

        self.assertTrue(move.destination.is_file())
        self.assertEqual((row.decision, row.action), ("non_target", "confirm"))

    def test_moving_into_the_current_folder_is_refused(self) -> None:
        move = record_decision(self.clip, "non_target", labeled_root=self.labeled_root, log=self.log)

        with self.assertRaisesRegex(ValueError, "confirm"):
            record_decision(move.destination, "non_target", labeled_root=self.labeled_root, log=self.log)
        self.assertEqual(len(self.log.rows()), 1)

    def test_existing_destination_is_never_overwritten(self) -> None:
        existing = self._write(self.labeled_root / "litoria_aurea" / self.clip.name, b"existing")

        with self.assertRaises(FileExistsError):
            record_decision(self.clip, "litoria_aurea", labeled_root=self.labeled_root, log=self.log)
        self.assertTrue(self.clip.exists())
        self.assertEqual(existing.read_bytes(), b"existing")
        self.assertFalse(self.log.path.exists())

    def test_faint_only_applies_to_frogs(self) -> None:
        with self.assertRaisesRegex(ValueError, "faint"):
            record_decision(self.clip, "non_target", labeled_root=self.labeled_root, log=self.log, faint=True)

    def test_confirm_refuses_a_clip_outside_the_labeled_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "labeled"):
            confirm_decision(self.clip, labeled_root=self.labeled_root, log=self.log)

    def test_undo_returns_the_clip_and_logs_where_it_went(self) -> None:
        move = record_decision(self.clip, "unsure", labeled_root=self.labeled_root, log=self.log)

        returned = undo_move(move, labeled_root=self.labeled_root, log=self.log)

        self.assertEqual(returned, self.clip)
        self.assertTrue(self.clip.is_file())
        self.assertFalse(move.destination.exists())
        last = self.log.rows()[-1]
        self.assertEqual((last.decision, last.action), ("queue", "undo"))

    def test_undo_of_a_change_logs_the_folder_it_returned_to(self) -> None:
        first = record_decision(self.clip, "non_target", labeled_root=self.labeled_root, log=self.log)
        second = record_decision(first.destination, "litoria_aurea", labeled_root=self.labeled_root, log=self.log)

        undo_move(second, labeled_root=self.labeled_root, log=self.log)

        self.assertEqual(self.log.rows()[-1].decision, "non_target")
        self.assertEqual(current_folder(first.destination, self.labeled_root), "non_target")

    def test_summarize_counts_clips_and_distinct_recordings(self) -> None:
        self._write(self.labeled_root / "litoria_aurea" / "20220101_220000_start5s.png")
        self._write(self.labeled_root / "litoria_aurea" / "20220101_220000_start10s.png")
        self._write(self.labeled_root / "non_target" / "20220102_100000_start0s.png")
        self._write(self.labeled_root / "unsure" / "20220103_010000_start0s.png")

        summary = summarize_labels(self.labeled_root)

        self.assertEqual((summary["litoria_aurea"].clips, summary["litoria_aurea"].recordings), (2, 1))
        self.assertEqual((summary["non_target"].clips, summary["non_target"].recordings), (1, 1))
        self.assertEqual((summary["unsure"].clips, summary["unsure"].recordings), (1, 1))

    def _write(self, path: Path, content: bytes = b"png") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.labeling.test_decisions -v`
Expected: FAIL at import with `ModuleNotFoundError: No module named 'frog_classifier.labeling.decisions'`.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/labeling/decisions.py`:

```python
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from frog_classifier.data.manifest import HOLDING_DIRECTORIES
from frog_classifier.data.naming import ExampleNameError, parse_example_filename


CLASS_FOLDERS = ("litoria_aurea", "non_target")
HOLDING_FOLDER = HOLDING_DIRECTORIES[0]
DECISION_FOLDERS = (*CLASS_FOLDERS, HOLDING_FOLDER)
LOG_COLUMNS = ("example_id", "decision", "action", "faint", "decided_at")
LOG_NAME = "decisions.csv"


@dataclass(frozen=True)
class DecisionRow:
    example_id: str
    decision: str
    action: str
    faint: bool
    decided_at: str


@dataclass(frozen=True)
class Move:
    source: Path
    destination: Path


@dataclass(frozen=True)
class ClassProgress:
    clips: int
    recordings: int


class DecisionLog:
    """Append-only record of every labeling decision. Nothing reads it to
    build training data; it exists so a changed mind leaves a trace."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, example_id: str, decision: str, action: str, *, faint: bool = False) -> DecisionRow:
        row = DecisionRow(example_id, decision, action, faint, _utc_now())
        is_new = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if is_new:
                writer.writerow(LOG_COLUMNS)
            writer.writerow([row.example_id, row.decision, row.action, "1" if row.faint else "0", row.decided_at])
        return row

    def rows(self) -> list[DecisionRow]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            return [
                DecisionRow(
                    record["example_id"],
                    record["decision"],
                    record["action"],
                    record["faint"] == "1",
                    record["decided_at"],
                )
                for record in csv.DictReader(handle)
            ]


def current_folder(image_path: Path, labeled_root: Path) -> str | None:
    """The decision folder holding the clip, or None when it is outside them."""
    try:
        relative = image_path.resolve().relative_to(labeled_root.resolve())
    except ValueError:
        return None
    if len(relative.parts) == 2 and relative.parts[0] in DECISION_FOLDERS:
        return relative.parts[0]
    return None


def record_decision(
    image_path: Path,
    decision: str,
    *,
    labeled_root: Path,
    log: DecisionLog,
    faint: bool = False,
    chunk_seconds: int = 5,
) -> Move:
    """Move the clip into the decision folder and log it. Never overwrites."""
    if decision not in DECISION_FOLDERS:
        raise ValueError(f"decision must be one of {DECISION_FOLDERS}")
    if faint and decision != "litoria_aurea":
        raise ValueError("faint applies to litoria_aurea decisions only")
    key = parse_example_filename(image_path, chunk_seconds=chunk_seconds)
    origin = current_folder(image_path, labeled_root)
    if origin == decision:
        raise ValueError("the clip is already in that folder; confirm it instead")
    destination = labeled_root / decision / image_path.name
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(image_path, destination)
    log.append(key.example_id, decision, "label" if origin is None else "change", faint=faint)
    return Move(source=image_path, destination=destination)


def confirm_decision(
    image_path: Path,
    *,
    labeled_root: Path,
    log: DecisionLog,
    chunk_seconds: int = 5,
) -> DecisionRow:
    """Log that an audited clip keeps its current folder. Moves nothing."""
    key = parse_example_filename(image_path, chunk_seconds=chunk_seconds)
    origin = current_folder(image_path, labeled_root)
    if origin is None:
        raise ValueError("only a clip inside the labeled folders can be confirmed")
    return log.append(key.example_id, origin, "confirm")


def undo_move(
    move: Move,
    *,
    labeled_root: Path,
    log: DecisionLog,
    chunk_seconds: int = 5,
) -> Path:
    """Return the clip to where it came from and log where it went."""
    if not move.destination.exists():
        raise FileNotFoundError(f"nothing to undo at {move.destination}")
    if move.source.exists():
        raise FileExistsError(f"refusing to overwrite {move.source}")
    move.source.parent.mkdir(parents=True, exist_ok=True)
    os.replace(move.destination, move.source)
    key = parse_example_filename(move.source, chunk_seconds=chunk_seconds)
    log.append(key.example_id, current_folder(move.source, labeled_root) or "queue", "undo")
    return move.source


def summarize_labels(labeled_root: Path, *, chunk_seconds: int = 5) -> dict[str, ClassProgress]:
    """Clip and distinct-recording counts per decision folder."""
    summary: dict[str, ClassProgress] = {}
    for folder in DECISION_FOLDERS:
        directory = labeled_root / folder
        paths = sorted(directory.glob("*.png")) if directory.is_dir() else []
        recordings: set[str] = set()
        for path in paths:
            try:
                recordings.add(parse_example_filename(path, chunk_seconds=chunk_seconds).recording_id)
            except ExampleNameError:
                continue
        summary[folder] = ClassProgress(clips=len(paths), recordings=len(recordings))
    return summary


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
```

Replace `src/frog_classifier/labeling/__init__.py` with:

```python
from .decisions import (
    CLASS_FOLDERS,
    DECISION_FOLDERS,
    HOLDING_FOLDER,
    LOG_COLUMNS,
    LOG_NAME,
    ClassProgress,
    DecisionLog,
    DecisionRow,
    Move,
    confirm_decision,
    current_folder,
    record_decision,
    summarize_labels,
    undo_move,
)
from .queue import NIGHT_HOURS, build_queue, hour_of_day, is_night

__all__ = [
    "CLASS_FOLDERS",
    "DECISION_FOLDERS",
    "HOLDING_FOLDER",
    "LOG_COLUMNS",
    "LOG_NAME",
    "NIGHT_HOURS",
    "ClassProgress",
    "DecisionLog",
    "DecisionRow",
    "Move",
    "build_queue",
    "confirm_decision",
    "current_folder",
    "hour_of_day",
    "is_night",
    "record_decision",
    "summarize_labels",
    "undo_move",
]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.labeling.test_decisions -v`
Expected: `Ran 11 tests`, OK.

- [ ] **Step 5: Full suite and commit**

```bash
git add src/frog_classifier/labeling tests/labeling/test_decisions.py
git diff --cached --check
git commit -m "feat: record labeling decisions and move clips without overwriting"
```

---

### Task 4: Playback audio lookup and export

**Files:**
- Create: `src/frog_classifier/labeling/audio.py`
- Modify: `src/frog_classifier/labeling/__init__.py`
- Create: `tests/labeling/test_audio.py`

**Interfaces:**
- Consumes: `PreprocessingConfig` (`config.audio.sample_rate_hz`, `config.audio.chunk_seconds`); test helpers `tone`, `silence`, `write_wav` from `tests/preprocessing/helpers.py`; `load_test_config` from `tests/data/helpers.py`.
- Produces: `find_recording(raw_root, recording_id, *, preferred_folder=None) -> Path | None`, `find_or_export_chunk(image_path, *, raw_root, chunk_root, config, spectrogram_root=None) -> Path | None`, `write_wav_mono_16bit(out_path, waveform, sample_rate_hz) -> None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/labeling/test_audio.py`:

```python
from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from frog_classifier.labeling.audio import find_or_export_chunk, find_recording
from tests.data.helpers import load_test_config
from tests.preprocessing.helpers import silence, tone, write_wav


class AudioTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = load_test_config(self.root)
        self.raw_root = self.root / "raw"
        self.chunk_root = self.root / "chunks"
        self.spectrogram_root = self.root / "spectrograms"
        write_wav(self.raw_root / "site" / "rec.wav", np.concatenate([silence(5), tone(1000.0, 5), silence(2)]))
        self.image = self._touch(self.spectrogram_root / "site" / "rec_start5s.png")

    def test_returns_the_cached_chunk_without_exporting(self) -> None:
        cached = self._touch(self.chunk_root / "site" / "rec_start5s.wav", b"cached")

        found = find_or_export_chunk(self.image, raw_root=self.raw_root, chunk_root=self.chunk_root, config=self.config, spectrogram_root=self.spectrogram_root)

        self.assertEqual(found, cached)
        self.assertEqual(cached.read_bytes(), b"cached")

    def test_exports_the_right_window_as_mono_16bit_wav(self) -> None:
        exported = find_or_export_chunk(self.image, raw_root=self.raw_root, chunk_root=self.chunk_root, config=self.config, spectrogram_root=self.spectrogram_root)

        self.assertEqual(exported, self.chunk_root / "site" / "rec_start5s.wav")
        with wave.open(str(exported), "rb") as handle:
            self.assertEqual((handle.getnchannels(), handle.getsampwidth(), handle.getframerate()), (1, 2, 22050))
            self.assertEqual(handle.getnframes(), 5 * 22050)
            samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
        self.assertGreater(int(np.abs(samples).max()), 10000)

    def test_exports_silence_for_a_silent_window(self) -> None:
        image = self._touch(self.spectrogram_root / "site" / "rec_start0s.png")

        exported = find_or_export_chunk(image, raw_root=self.raw_root, chunk_root=self.chunk_root, config=self.config, spectrogram_root=self.spectrogram_root)

        with wave.open(str(exported), "rb") as handle:
            samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
        self.assertEqual(int(np.abs(samples).max()), 0)

    def test_labeled_clip_finds_the_recording_without_a_spectrogram_root(self) -> None:
        labeled = self._touch(self.root / "labeled" / "non_target" / "rec_start5s.png")

        exported = find_or_export_chunk(labeled, raw_root=self.raw_root, chunk_root=self.chunk_root, config=self.config)

        self.assertEqual(exported, self.chunk_root / "site" / "rec_start5s.wav")

    def test_missing_or_ambiguous_recording_returns_none(self) -> None:
        ghost = self._touch(self.spectrogram_root / "site" / "ghost_start0s.png")
        self.assertIsNone(find_or_export_chunk(ghost, raw_root=self.raw_root, chunk_root=self.chunk_root, config=self.config))

        self._touch(self.raw_root / "elsewhere" / "rec.mp3")
        self.assertIsNone(find_recording(self.raw_root, "rec"))

    def test_preferred_folder_wins_over_a_duplicate_elsewhere(self) -> None:
        self._touch(self.raw_root / "elsewhere" / "rec.mp3")

        found = find_recording(self.raw_root, "rec", preferred_folder=Path("site"))

        self.assertEqual(found, self.raw_root / "site" / "rec.wav")

    def _touch(self, path: Path, content: bytes = b"") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.labeling.test_audio -v`
Expected: FAIL at import with `ModuleNotFoundError: No module named 'frog_classifier.labeling.audio'`.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/labeling/audio.py`:

```python
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from frog_classifier.data.config import PreprocessingConfig
from frog_classifier.data.naming import ExampleNameError, parse_example_filename


SUPPORTED_EXTS = {".wav", ".mp3"}


def find_recording(
    raw_root: Path,
    recording_id: str,
    *,
    preferred_folder: Path | None = None,
) -> Path | None:
    """The recording with this stem: the preferred folder first, then a
    recursive search that must find exactly one file."""
    if preferred_folder is not None:
        folder = raw_root / preferred_folder
        if folder.is_dir():
            direct = sorted(path for path in folder.iterdir() if _is_recording(path, recording_id))
            if len(direct) == 1:
                return direct[0]
    matches = sorted(path for path in raw_root.rglob("*") if _is_recording(path, recording_id))
    return matches[0] if len(matches) == 1 else None


def find_or_export_chunk(
    image_path: Path,
    *,
    raw_root: Path,
    chunk_root: Path,
    config: PreprocessingConfig,
    spectrogram_root: Path | None = None,
) -> Path | None:
    """The playback WAV for a clip, exported from the recording on first use.

    The chunk lives under chunk_root in the recording's folder relative to
    raw_root, so a clip found in the queue and the same clip after labeling
    share one cached file.
    """
    try:
        key = parse_example_filename(image_path, chunk_seconds=config.audio.chunk_seconds)
    except ExampleNameError:
        return None
    preferred = None
    if spectrogram_root is not None:
        try:
            preferred = image_path.resolve().relative_to(spectrogram_root.resolve()).parent
        except ValueError:
            preferred = None
    recording = find_recording(raw_root, key.recording_id, preferred_folder=preferred)
    if recording is None:
        return None
    relative_folder = recording.resolve().parent.relative_to(raw_root.resolve())
    chunk_path = chunk_root / relative_folder / f"{key.example_id}.wav"
    if chunk_path.is_file():
        return chunk_path

    import librosa  # imported lazily so the labeler starts quickly

    waveform, sample_rate = librosa.load(
        recording,
        sr=config.audio.sample_rate_hz,
        mono=True,
        offset=float(key.start_s),
        duration=float(config.audio.chunk_seconds),
    )
    if len(waveform) == 0:
        return None
    write_wav_mono_16bit(chunk_path, waveform, int(sample_rate))
    return chunk_path


def write_wav_mono_16bit(out_path: Path, waveform: np.ndarray, sample_rate_hz: int) -> None:
    """Mono 16-bit PCM WAV through the standard library."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.clip(np.asarray(waveform, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    with wave.open(str(out_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm.tobytes())


def _is_recording(path: Path, recording_id: str) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTS and path.stem == recording_id
```

In `src/frog_classifier/labeling/__init__.py` add, as the first import block:

```python
from .audio import find_or_export_chunk, find_recording, write_wav_mono_16bit
```

and add `"find_or_export_chunk",`, `"find_recording",`, and `"write_wav_mono_16bit",` to `__all__` in alphabetical position.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.labeling.test_audio -v`
Expected: `Ran 6 tests`, OK, no warnings (the audioread shim import in `tests/__init__.py` already silences librosa's backend probe).

- [ ] **Step 5: Full suite and commit**

```bash
git add src/frog_classifier/labeling tests/labeling/test_audio.py
git diff --cached --check
git commit -m "feat: locate or export playback audio through the labeling package"
```

---

### Task 5: Streamlit labeler with audit mode, terminal labeler retired

**Files:**
- Modify: `scripts/label_frontend.py` (rewrite everything below `_inject_sidebar_css`, keep that function and the imports it needs)
- Delete: `scripts/label_spectrograms.py`
- Modify: `tests/test_repository_reproducibility.py`

**Interfaces:**
- Consumes: everything exported from `frog_classifier.labeling`, `colorize` from `frog_classifier.preprocessing.display`, `load_preprocessing_config` from `frog_classifier.data`.

- [ ] **Step 1: Update the reproducibility test so it fails now**

In `tests/test_repository_reproducibility.py`:
- Add after `PREPROCESSING_PACKAGE_MODULES`:

```python
LABELING_PACKAGE_MODULES = (
    "src/frog_classifier/labeling/__init__.py",
    "src/frog_classifier/labeling/audio.py",
    "src/frog_classifier/labeling/decisions.py",
    "src/frog_classifier/labeling/queue.py",
)
```

- Add `*LABELING_PACKAGE_MODULES,` to the loop in `test_project_navigation_files_are_complete_and_linked` after `*PREPROCESSING_PACKAGE_MODULES,`.
- Add `"scripts/label_spectrograms.py",` to `OBSOLETE_PATHS`.
- Remove `"scripts/label_spectrograms.py",` from `OPERATIONAL_TEXT_FILES`.

Run: `.venv/Scripts/python.exe -m unittest tests.test_repository_reproducibility -v`
Expected: FAIL because `scripts/label_spectrograms.py` still exists.

- [ ] **Step 2: Rewrite the labeler**

In `scripts/label_frontend.py`, replace the import block at the top with:

```python
from __future__ import annotations

from pathlib import Path

import streamlit as st

from frog_classifier.data import load_preprocessing_config
from frog_classifier.labeling import (
    DECISION_FOLDERS,
    LOG_NAME,
    DecisionLog,
    build_queue,
    confirm_decision,
    current_folder,
    find_or_export_chunk,
    record_decision,
    summarize_labels,
    undo_move,
)
from frog_classifier.preprocessing.display import colorize


REPO_ROOT = Path(__file__).resolve().parents[1]
MODE_LABEL = "Label new clips"
MODE_AUDIT = "Audit labeled clips"
BUTTONS = (
    ("Frog (target)", "litoria_aurea", False),
    ("Frog, faint", "litoria_aurea", True),
    ("Background (no target frog)", "non_target", False),
    ("Unsure (review later)", "unsure", False),
)
```

Keep `_inject_sidebar_css` exactly as it is. Delete every other function (`_repo_root`, `_candidate_dirs`, `_iter_pngs`, `_read_bytes`, `_count_pngs`, `_safe_move`, `_label_and_track`, `main`) and replace them with:

```python
def _pngs_under(root: Path) -> list[Path]:
    return sorted(root.rglob("*.png")) if root.is_dir() else []


def _labeled_pngs(labeled_root: Path) -> list[Path]:
    clips: list[Path] = []
    for folder in DECISION_FOLDERS:
        directory = labeled_root / folder
        if directory.is_dir():
            clips.extend(sorted(directory.glob("*.png")))
    return clips


def _reset_session() -> None:
    for key in ("queue", "index", "last_move", "error"):
        st.session_state.pop(key, None)


def _decide(clip: Path, decision: str, faint: bool, *, labeled_root: Path, log: DecisionLog, chunk_seconds: int) -> None:
    try:
        if current_folder(clip, labeled_root) == decision:
            confirm_decision(clip, labeled_root=labeled_root, log=log, chunk_seconds=chunk_seconds)
            st.session_state["last_move"] = None
        else:
            st.session_state["last_move"] = record_decision(
                clip, decision, labeled_root=labeled_root, log=log, faint=faint, chunk_seconds=chunk_seconds,
            )
    except (ValueError, FileExistsError) as error:
        st.session_state["error"] = str(error)
        return
    st.session_state.pop("error", None)
    st.session_state["index"] += 1


def main() -> None:
    st.set_page_config(page_title="Frog Spectrogram Labeler", layout="wide")
    _inject_sidebar_css()
    st.title("Frog Spectrogram Labeler")
    config = load_preprocessing_config(REPO_ROOT / "config" / "preprocessing.toml")
    chunk_seconds = config.audio.chunk_seconds

    with st.sidebar:
        mode = st.radio("Mode", (MODE_LABEL, MODE_AUDIT), on_change=_reset_session)
        st.header("Quick start")
        st.write("1) Press Play and listen")
        st.write("2) Choose Frog, Frog faint, Background, or Unsure")
        st.write("3) In audit mode, pressing the current label confirms it")
        with st.expander("Advanced settings", expanded=False):
            spectrogram_root = Path(st.text_input("spectrogram_root", str(REPO_ROOT / "processed" / "external" / "spectrograms")))
            raw_root = Path(st.text_input("raw_root", str(REPO_ROOT / "raw" / "external")))
            chunk_root = Path(st.text_input("chunk_root", str(REPO_ROOT / "processed" / "external" / "chunks")))
            labeled_root = Path(st.text_input("labeled_root", str(REPO_ROOT / "labeled")))
            seed = int(st.number_input("Seed (queue order)", min_value=0, max_value=10_000_000, value=1337, step=1))
            limit = int(st.number_input("Limit (0 = all)", min_value=0, value=0, step=1))
            cap = int(st.number_input("Clips per recording per session", min_value=1, max_value=100, value=5, step=1))
        st.divider()
        if st.button("Reload clips", use_container_width=True):
            _reset_session()
            st.rerun()

    log = DecisionLog(labeled_root / LOG_NAME)

    if "queue" not in st.session_state:
        source = _pngs_under(spectrogram_root) if mode == MODE_LABEL else _labeled_pngs(labeled_root)
        queue = build_queue(source, seed=seed, chunk_seconds=chunk_seconds, per_recording_cap=cap)
        if limit > 0:
            queue = queue[:limit]
        st.session_state.update(queue=queue, index=0, last_move=None)

    queue: list[Path] = st.session_state["queue"]
    index: int = st.session_state["index"]

    progress = summarize_labels(labeled_root, chunk_seconds=chunk_seconds)
    columns = st.columns(4)
    for column, folder, title in zip(columns, DECISION_FOLDERS, ("Frog", "Background", "Unsure")):
        column.metric(title, progress[folder].clips, f"{progress[folder].recordings} recordings")
    columns[3].metric("Remaining (this session)", max(len(queue) - index, 0))

    if st.session_state.get("error"):
        st.error(st.session_state["error"])

    if not queue:
        st.warning("No clips found for this mode.")
        st.stop()
    if index >= len(queue):
        st.success("Done. No more clips in this session.")
        st.stop()

    clip = queue[index]
    st.subheader(f"Clip {index + 1} of {len(queue)}")
    st.progress((index + 1) / len(queue))
    st.caption(str(clip))
    if mode == MODE_AUDIT:
        st.info(f"Current label: {current_folder(clip, labeled_root)}")

    col_img, col_controls = st.columns([2, 1], gap="large")
    with col_img:
        st.image(colorize(clip.read_bytes()), use_container_width=True)

    audio_path = find_or_export_chunk(
        clip, raw_root=raw_root, chunk_root=chunk_root, config=config,
        spectrogram_root=spectrogram_root if mode == MODE_LABEL else None,
    )

    with col_controls:
        st.markdown("### Actions")
        if audio_path is not None:
            st.audio(audio_path.read_bytes(), format="audio/wav")
        else:
            st.info("No audio found for this clip.")
        for label, decision, faint in BUTTONS:
            kind = "primary" if decision == "litoria_aurea" and not faint else "secondary"
            if st.button(label, type=kind, use_container_width=True):
                _decide(clip, decision, faint, labeled_root=labeled_root, log=log, chunk_seconds=chunk_seconds)
                st.rerun()
        left, right = st.columns(2, gap="small")
        with left:
            if st.button("Skip", use_container_width=True):
                st.session_state["index"] += 1
                st.rerun()
        with right:
            if st.button("Undo last move", use_container_width=True):
                move = st.session_state.get("last_move")
                if move is None:
                    st.session_state["error"] = "Nothing to undo."
                else:
                    try:
                        undo_move(move, labeled_root=labeled_root, log=log, chunk_seconds=chunk_seconds)
                        st.session_state["last_move"] = None
                        st.session_state["index"] = max(index - 1, 0)
                        st.session_state.pop("error", None)
                    except (FileNotFoundError, FileExistsError) as error:
                        st.session_state["error"] = str(error)
                st.rerun()
        st.divider()
        if st.button("End session", use_container_width=True):
            st.stop()


if __name__ == "__main__":
    main()
```

Delete `scripts/label_spectrograms.py` with `git rm scripts/label_spectrograms.py`.

- [ ] **Step 3: Compile and run the reproducibility test**

Run: `.venv/Scripts/python.exe -B -m py_compile scripts/label_frontend.py`
Expected: no output.

Run: `.venv/Scripts/python.exe -m unittest tests.test_repository_reproducibility -v`
Expected: OK.

- [ ] **Step 4: Manual check on synthetic data (no project data)**

Build a throwaway tree with one recording and its images, then drive the labeler:

```bash
.venv/Scripts/python.exe -c "
from pathlib import Path
import numpy as np
from frog_classifier.data import load_preprocessing_config
from frog_classifier.preprocessing import render_waveform
from tests.preprocessing.helpers import silence, tone, write_wav
root = Path('.scratch/labeler'); (root / 'labeled').mkdir(parents=True, exist_ok=True)
config = load_preprocessing_config(Path('config/preprocessing.toml'))
wave = np.concatenate([silence(5), tone(1000.0, 5), silence(5)])
write_wav(root / 'raw' / 'site' / '20220101_220000.wav', wave)
out = root / 'spectrograms' / 'site'; out.mkdir(parents=True, exist_ok=True)
for chunk in render_waveform(wave, config):
    (out / f'20220101_220000_start{chunk.start_s}s.png').write_bytes(chunk.png_bytes)
print('ready')
"
.venv/Scripts/python.exe -m streamlit run scripts/label_frontend.py --server.address localhost --server.port 8501 --server.headless true
```

In the browser set the four paths under Advanced settings to the absolute paths of `.scratch/labeler/spectrograms`, `.scratch/labeler/raw`, `.scratch/labeler/chunks`, and `.scratch/labeler/labeled`, press Reload clips, label one clip Frog faint, one Background, one Unsure, press Undo once, then switch to audit mode and confirm one clip and change another. Confirm `.scratch/labeler/labeled/decisions.csv` has one row per press with the expected actions, and the files sit in the expected folders. Stop Streamlit and delete `.scratch/`.

If the controller has ruled that the browser check is theirs, skip the browser part and only run the data-building command and `py_compile`.

- [ ] **Step 5: Full suite and commit**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: OK, no warnings.

```bash
git add scripts/label_frontend.py tests/test_repository_reproducibility.py
git diff --cached --check
git commit -m "feat: add audit mode and unsure state to the labeler and retire the terminal labeler"
```

---

### Task 6: Documentation

**Files:**
- Modify: `README.md`, `AGENTS.md`, `docs/outline.md`, `docs/architecture.md`, `docs/workflow.md`, `roadmap.md`
- Modify: `tests/test_repository_reproducibility.py`

- [ ] **Step 1: Extend the documentation test so it fails now**

Add to `OperationalDocumentationTests` in `tests/test_repository_reproducibility.py`:

```python
    def test_documents_never_mention_the_retired_terminal_labeler(self) -> None:
        for relative_path in ("README.md", "AGENTS.md", "docs/outline.md", "docs/architecture.md", "docs/workflow.md"):
            text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
            with self.subTest(file=relative_path):
                self.assertNotIn("label_spectrograms.py", text)
                self.assertNotIn("Use Skip when identification is uncertain", text)
```

Run: `.venv/Scripts/python.exe -m unittest tests.test_repository_reproducibility -v`
Expected: FAIL on README, outline, architecture, and workflow.

- [ ] **Step 2: README**

Remove the block:

````markdown
Launch the terminal labeler:

```powershell
uv run python scripts/label_spectrograms.py --limit 12 --shuffle
```

````

Replace the "Labeling policy" section with:

```markdown
## Labeling policy

Use Frog only when the target call is confidently present, and add the faint tag when it is distant.
Use Background only when the clip is confidently non-target.
Use Unsure when you cannot decide; the clip waits in `labeled/unsure/` and never enters training.
Every decision is appended to `labeled/decisions.csv`, and the labeler's audit mode replays labeled clips so a decision can be confirmed or changed.
```

- [ ] **Step 3: AGENTS.md**

Replace `- Skip uncertain examples instead of converting uncertainty into a negative label.` with `- Mark uncertain examples Unsure instead of converting uncertainty into a negative label; unsure clips never enter training.`

Replace `- \`labeled/\` contains human-selected examples and the generated manifest.` with `- \`labeled/\` contains human-selected examples, the unsure holding folder, the decision log, and the generated manifest.`

- [ ] **Step 4: Outline**

In the repository structure block: remove the `label_spectrograms.py` line; change the `label_frontend.py` description to `Streamlit labeler with label and audit modes`; under `labeled/` add `unsure/                    clips the reviewer could not decide on` and `decisions.csv              append-only log of every labeling decision` after the `non_target/` line; under `src/frog_classifier/` add `labeling/                  queue order, decision log, moves, and playback audio` after the `preprocessing/` line.

Replace the "3. Human labeling" section with:

```markdown
### 3. Human labeling

The Streamlit labeler shows each spectrogram with its matching audio window.
A confident target call moves to `labeled/litoria_aurea`, a confident background example moves to `labeled/non_target`, and a clip you cannot decide on moves to `labeled/unsure`.
Every press is appended to `labeled/decisions.csv`, and the audit mode replays labeled clips so a decision can be confirmed or changed later.
```

In "Label examples", remove the sentence introducing the terminal labeler and its code block, and replace `Use Skip when identification is uncertain, and never convert uncertainty into a negative label.` with `Use Unsure when identification is uncertain, and never convert uncertainty into a negative label.`

Replace `Skip uncertain clips until Phase 3 adds an explicit review-later state and signal-quality metadata.` with `Mark uncertain clips Unsure so they wait for a second hearing instead of becoming background labels.`

In the `py_compile` command, remove `scripts/label_spectrograms.py`.

In the "Where to make a change" table add `| Labeling queue, decisions, or playback audio | \`src/frog_classifier/labeling/\` |` after the spectrogram rendering row.

- [ ] **Step 5: Architecture**

In the storage contract block add `labeled/unsure/              clips the reviewer could not decide on, excluded from training`, `labeled/decisions.csv        append-only log of labeling decisions`, and `src/frog_classifier/labeling/   session queue, decision log, moves, playback audio` in the natural positions.

Replace the "Human labeling" section with:

```markdown
## Human labeling

`scripts/label_frontend.py` is the only labeler; `Launch_Labeler.bat` starts it.
It shows a spectrogram through the viridis colour map with its five-second audio window, which `frog_classifier.labeling.audio` locates in the playback cache or exports from the recording on first use.
In label mode the queue comes from `frog_classifier.labeling.queue`: night clips (19:00 to 06:59) shuffled with a seed, at most five per recording per session, with one daytime clip for every nine night clips.
In audit mode the queue is the labeled folders in the same order, the current class is shown, pressing it confirms, and pressing another class moves the clip.
Frog, Frog faint, Background, and Unsure move the PNG into `labeled/litoria_aurea`, `labeled/non_target`, or `labeled/unsure` through `frog_classifier.labeling.decisions`, which refuses to overwrite and appends one row per press to `labeled/decisions.csv` with the resulting folder, the action (`label`, `confirm`, `change`, or `undo`), the faint flag, and a UTC timestamp.
Skip leaves no record, and Undo reverts only the last move of the session.
The manifest builder skips `labeled/unsure/`, so unsure clips never enter training.
```

- [ ] **Step 6: Workflow**

Replace the "Apply labels" section, including "Labeling policy", with:

````markdown
## Apply labels

Double-click `Launch_Labeler.bat`, or run the Streamlit labeler directly:

```powershell
uv run python -m streamlit run scripts/label_frontend.py
```

The sidebar switches between two modes.
"Label new clips" walks the queue under the spectrogram root: night clips first, shuffled with the seed, at most five per recording per session, with one daytime clip for every nine night clips.
"Audit labeled clips" walks the three labeled folders in the same order, shows the current label, and lets you confirm it by pressing the same button or change it by pressing another.
Both modes play the five-second audio window, exporting it from the recording on first use.

Buttons: Frog, Frog faint, Background, Unsure, Skip, and Undo.
Skip leaves no record.
Undo reverts only the last move of the session.
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
````

In the `py_compile` command, remove `scripts/label_spectrograms.py`.

- [ ] **Step 7: Roadmap**

Under "Phase 3: Purposeful label expansion", after the "### Work" list, add:

```markdown
### Progress

September 13, 2026: the labeler gained an Unsure state, an append-only decision log, an audit mode, and a night-first queue with a per-recording cap.
The audit of the existing 61 labels is the next labeling session.
```

In the "Current checkpoint" list add `- Unsure \`labeled/unsure\` clips: 0` after the negative labels line.

- [ ] **Step 8: Run the checks and commit**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: OK.

Run: `grep -nP '\x{2014}' README.md AGENTS.md docs/outline.md docs/architecture.md docs/workflow.md roadmap.md`
Expected: no output.

Run: `git diff --check`
Expected: no output.

```bash
git add README.md AGENTS.md docs/outline.md docs/architecture.md docs/workflow.md roadmap.md tests/test_repository_reproducibility.py
git diff --cached --check
git commit -m "docs: describe the unsure state, decision log, and audit mode"
```

---

## Self-review against the spec

- Unsure holding folder ignored by the manifest, tracked placeholder: Task 1.
- Queue: night hours, shuffle, cap, nine-to-one interleave, timestamp-less IDs as daytime: Task 2.
- Decision log columns and semantics, moves that never overwrite, confirm, change, undo, progress: Task 3.
- Audio lookup with mirrored folder then recursive search, chunk path from the recording folder, export as mono 16-bit WAV: Task 4.
- Streamlit modes, buttons, error display, progress row, terminal labeler deleted, reproducibility test: Task 5.
- Documentation and roadmap: Task 6.
- Not touching manifest schema, reports, sync, verify, preprocessing: no task does.
