# Batch Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a saved model over whole folders of recordings and keep a resumable, traceable record of every window's frog probability.

**Architecture:** A new `frog_classifier.inference` package holds recording discovery and start times, worker-side rendering without PyTorch, the SQLite detection store, the CSV export, and the detection run that ties them together. The preprocessing package exposes window images before PNG encoding so slicing and detection share pixels, and the training package gains one saved-run loader used by both verification and detection. `scripts/detect_calls.py` wraps it with a progress bar.

**Tech Stack:** Python 3.13 standard library (`sqlite3`, `concurrent.futures`, `multiprocessing`), librosa, PyTorch 2.14, tqdm, unittest.

**Spec:** `docs/superpowers/specs/2026-10-15-batch-detection-design.md`

## Global Constraints

- No new dependencies: `pyproject.toml`, `uv.lock`, and `requirements_labeler.txt` do not change.
- uv is at `/c/Users/ghimi/.local/bin/uv.exe`; it is not on PATH in this shell. Run Python as `.venv/Scripts/python.exe`.
- Text files use LF line endings; write files with the editor tools or bytes, never Python `write_text` on Windows.
- Never use the em dash character anywhere. Each full sentence in long Markdown files sits on its own line.
- Commit messages carry no co-author trailer.
- Tests use synthetic audio in temporary folders, the CPU, and no downloads; no test reads `raw/`, `labeled/`, or `models/`.
- `frog_classifier.inference.recordings`, `frog_classifier.inference.rendering`, and everything they import never import PyTorch.
- Detection output defaults to `results/detections/`, which Git ignores. Never commit `labeled/`, `models/`, or `results/` contents.
- Raw recordings are read only; nothing is ever written inside an input folder.

## Review Focus

- The same file name in two folders is two recordings, stored and exported separately by relative path (Task 7 test data).
- File names with spaces or non-ASCII characters are found, stored, and exported intact (Task 3 discovery test, Task 6 export data).
- Interrupting a run that uses worker processes stops promptly and leaves a record the next run resumes (Task 7 interruption test with two workers).
- A rerun after everything is done processes nothing new and still rewrites the CSV files (Task 8 command test).
- A file that disappears between discovery and rendering is that recording's failure, not a crash (Task 4 test).

---

### Task 1: Window images shared by slicing and detection

**Files:**
- Modify: `src/frog_classifier/preprocessing/recording.py`
- Modify: `src/frog_classifier/preprocessing/__init__.py`
- Test: `tests/preprocessing/test_recording.py`

**Interfaces:**
- Produces: `ChunkImage(start_s: int, pixels: np.ndarray)` and `render_waveform_images(waveform: np.ndarray, config: PreprocessingConfig) -> Iterator[ChunkImage]`, both exported from `frog_classifier.preprocessing`. `render_waveform` keeps its signature and output.

- [ ] **Step 1: Write the failing test**

In `tests/preprocessing/test_recording.py`, extend the imports:

```python
from frog_classifier.preprocessing.recording import (
    RenderedChunk,
    render_recording,
    render_waveform,
    render_waveform_images,
)
from frog_classifier.preprocessing.spectrogram import decode_png
from tests.data.helpers import load_test_config
from tests.preprocessing.helpers import noise, silence, tone, write_wav
```

Add to `RenderRecordingTests`:

```python
    def test_window_images_are_the_pixels_each_png_stores(self) -> None:
        waveform = np.concatenate([noise(5), noise(5, seed=1) + tone(1000.0, 5), noise(6, seed=2)])

        images = list(render_waveform_images(waveform, self.config))
        chunks = list(render_waveform(waveform, self.config))

        self.assertEqual([image.start_s for image in images], [0, 5, 10])
        for image, chunk in zip(images, chunks, strict=True):
            self.assertEqual(image.pixels.dtype, np.uint8)
            np.testing.assert_array_equal(image.pixels, decode_png(chunk.png_bytes))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m unittest tests.preprocessing.test_recording -v`
Expected: ERROR with `ImportError: cannot import name 'render_waveform_images'`.

- [ ] **Step 3: Implement**

In `src/frog_classifier/preprocessing/recording.py`, add after `RenderedChunk`:

```python
@dataclass(frozen=True)
class ChunkImage:
    start_s: int
    pixels: np.ndarray
```

Replace `render_waveform` with:

```python
def render_waveform_images(
    waveform: np.ndarray,
    config: PreprocessingConfig,
) -> Iterator[ChunkImage]:
    """Yield one 8-bit spectrogram image per complete window, in order.

    The noise floor comes from every frame of the whole waveform, including
    any incomplete tail. Each chunk's spectrogram is computed from the chunk's
    own samples so that a batch or streaming inference path renders the same
    pixels.
    """
    samples_per_chunk = config.audio.sample_rate_hz * config.audio.chunk_seconds
    if len(waveform) < samples_per_chunk:
        return
    floor = noise_floor(
        _mel_decibels(waveform, config),
        config.normalization.noise_floor_percentile,
    )
    complete_chunks = len(waveform) // samples_per_chunk
    for index in range(complete_chunks):
        start = index * samples_per_chunk
        chunk = waveform[start : start + samples_per_chunk]
        yield ChunkImage(
            start_s=index * config.audio.chunk_seconds,
            pixels=chunk_image(
                _mel_decibels(chunk, config),
                floor,
                db_floor=config.normalization.db_floor,
                db_ceiling=config.normalization.db_ceiling,
            ),
        )


def render_waveform(
    waveform: np.ndarray,
    config: PreprocessingConfig,
) -> Iterator[RenderedChunk]:
    """Yield one PNG-encoded chunk per complete window, in order."""
    for image in render_waveform_images(waveform, config):
        yield RenderedChunk(start_s=image.start_s, png_bytes=encode_png(image.pixels))
```

In `src/frog_classifier/preprocessing/__init__.py`, import and export `ChunkImage` and `render_waveform_images` beside the existing names, keeping `__all__` sorted:

```python
from .recording import (
    ChunkImage,
    RenderedChunk,
    load_waveform,
    render_recording,
    render_waveform,
    render_waveform_images,
)
```

```python
__all__ = [
    "ChunkImage",
    "RenderedChunk",
    "chunk_image",
    "decode_png",
    "encode_png",
    "load_waveform",
    "mel_decibels",
    "noise_floor",
    "render_recording",
    "render_waveform",
    "render_waveform_images",
]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.preprocessing.test_recording tests.test_slice_audio -v`
Expected: every test passes, including the five recording tests.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/preprocessing/recording.py src/frog_classifier/preprocessing/__init__.py tests/preprocessing/test_recording.py
git commit -m "feat: expose window images before PNG encoding"
```

---

### Task 2: One saved-run loader for verification and detection

**Files:**
- Create: `src/frog_classifier/training/saved.py`
- Modify: `src/frog_classifier/training/artifact.py`
- Modify: `src/frog_classifier/training/run.py`
- Modify: `src/frog_classifier/training/verify.py`
- Modify: `tests/test_repository_reproducibility.py`
- Test: `tests/training/test_saved.py`

**Interfaces:**
- Produces: `PREPROCESSING_FILE = "preprocessing.toml"` in `frog_classifier.training.artifact`; in `frog_classifier.training.saved`: `SavedRunError(ValueError)`, `SavedRun(name: str, model: SpectrogramResNet, model_sha256: str, threshold: float, preprocessing: PreprocessingConfig)`, `run_problems(run_dir: Path, metadata: Mapping[str, Any]) -> list[str]`, `load_model(run_dir: Path) -> SpectrogramResNet` (CPU, evaluation mode), `load_saved_run(run_dir: Path) -> SavedRun`.

- [ ] **Step 1: Write the failing tests**

Create `tests/training/test_saved.py`:

```python
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from frog_classifier.training.artifact import MODEL_FILE, RUN_METADATA, file_sha256, read_run, write_json
from frog_classifier.training.run import train_and_save
from frog_classifier.training.saved import SavedRunError, load_saved_run
from tests.training.helpers import build_synthetic_repo


CPU = torch.device("cpu")


class LoadSavedRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.source = Path(cls.directory.name) / "source"
        paths = build_synthetic_repo(cls.source)
        cls.run_name = train_and_save(paths, cross_validate=False, device=CPU, progress=lambda _: None).name

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def setUp(self) -> None:
        # Each test gets its own copy so tampering never leaks into another test.
        working = tempfile.TemporaryDirectory()
        self.addCleanup(working.cleanup)
        self.repo = Path(working.name) / "repo"
        shutil.copytree(self.source, self.repo)
        self.run_dir = self.repo / "models" / self.run_name

    def test_loads_the_model_threshold_and_training_preprocessing(self) -> None:
        saved = load_saved_run(self.run_dir)

        metadata = read_run(self.run_dir)
        self.assertEqual(saved.name, self.run_name)
        self.assertEqual(saved.threshold, metadata["threshold"]["value"])
        self.assertEqual(saved.model_sha256, file_sha256(self.run_dir / MODEL_FILE))
        self.assertEqual(saved.preprocessing.sha256, metadata["checksums"]["preprocessing_config"])
        self.assertFalse(saved.model.training)

    def test_a_changed_model_file_is_refused(self) -> None:
        model = self.run_dir / MODEL_FILE
        model.write_bytes(model.read_bytes() + b"\0")

        with self.assertRaisesRegex(SavedRunError, "model.pt: checksum differs from run.json"):
            load_saved_run(self.run_dir)

    def test_an_unknown_architecture_is_refused(self) -> None:
        metadata = read_run(self.run_dir)
        metadata["model"]["architecture"] = "resnet50"
        write_json(self.run_dir / RUN_METADATA, metadata)

        with self.assertRaisesRegex(SavedRunError, "unsupported architecture 'resnet50'"):
            load_saved_run(self.run_dir)

    def test_a_folder_that_is_not_a_run_is_refused(self) -> None:
        with self.assertRaisesRegex(SavedRunError, "is not a saved run"):
            load_saved_run(self.repo / "labeled")


if __name__ == "__main__":
    unittest.main()
```

In `tests/test_repository_reproducibility.py`, add `"saved"` to `TRAINING_PACKAGE_MODULES`:

```python
TRAINING_PACKAGE_MODULES = tuple(
    f"src/frog_classifier/training/{name}.py"
    for name in (
        "__init__", "artifact", "augment", "baseline", "config", "fit",
        "images", "metrics", "model", "report", "run", "saved", "verify",
    )
)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.training.test_saved tests.test_repository_reproducibility -v`
Expected: `ModuleNotFoundError: No module named 'frog_classifier.training.saved'` and a failing `required=src/frog_classifier/training/saved.py` subtest.

- [ ] **Step 3: Implement**

In `src/frog_classifier/training/artifact.py`, add after `PREDICTIONS_FILE`:

```python
PREPROCESSING_FILE = "preprocessing.toml"
```

In `src/frog_classifier/training/run.py`, import `PREPROCESSING_FILE` from `.artifact` alongside the existing names and replace the literal key `"preprocessing.toml": preprocessing.source_path.read_bytes(),` with:

```python
        PREPROCESSING_FILE: preprocessing.source_path.read_bytes(),
```

Create `src/frog_classifier/training/saved.py`:

```python
"""Load a saved run for prediction, refusing one that is incomplete or altered."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from frog_classifier.data.config import PreprocessingConfig, load_preprocessing_config

from .artifact import (
    ARTIFACT_SCHEMA_VERSION,
    MODEL_FILE,
    PREPROCESSING_FILE,
    RUN_METADATA,
    check_files,
    read_run,
)
from .model import ARCHITECTURE, SpectrogramResNet


class SavedRunError(ValueError):
    pass


@dataclass(frozen=True)
class SavedRun:
    name: str
    model: SpectrogramResNet
    model_sha256: str
    threshold: float
    preprocessing: PreprocessingConfig


def run_problems(run_dir: Path, metadata: Mapping[str, Any]) -> list[str]:
    """Reasons a run cannot be trusted: a format or architecture this code does
    not know, or a file missing or changed since run.json recorded it."""
    version = metadata.get("artifact_schema_version")
    if version != ARTIFACT_SCHEMA_VERSION:
        return [f"unsupported run format {version!r}"]
    problems = check_files(run_dir, metadata)
    architecture = metadata["model"]["architecture"]
    if architecture != ARCHITECTURE:
        problems.append(f"unsupported architecture {architecture!r}")
    return problems


def load_model(run_dir: Path) -> SpectrogramResNet:
    """The run's trained network on the CPU, in evaluation mode."""
    model = SpectrogramResNet(pretrained=False)
    model.load_state_dict(torch.load(run_dir / MODEL_FILE, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_saved_run(run_dir: Path) -> SavedRun:
    """The run's model and threshold, with the preprocessing contract it was trained on.

    The contract comes from the run's own copy, so a later change to the tracked
    configuration can never change what this model is shown.
    """
    if not (run_dir / RUN_METADATA).is_file():
        raise SavedRunError(f"{run_dir} is not a saved run: it has no {RUN_METADATA}")
    metadata = read_run(run_dir)
    problems = run_problems(run_dir, metadata)
    if not problems:
        problems = [
            f"{name}: not recorded in {RUN_METADATA}"
            for name in (MODEL_FILE, PREPROCESSING_FILE)
            if name not in metadata["files"]
        ]
    if problems:
        raise SavedRunError(f"run {run_dir.name} cannot be used: " + "; ".join(problems))
    return SavedRun(
        name=metadata["run_name"],
        model=load_model(run_dir),
        model_sha256=metadata["files"][MODEL_FILE],
        threshold=float(metadata["threshold"]["value"]),
        preprocessing=load_preprocessing_config(run_dir / PREPROCESSING_FILE),
    )
```

In `src/frog_classifier/training/verify.py`, replace the imports of `.artifact` and `.model` with:

```python
from .artifact import PREDICTIONS_FILE, file_sha256, read_csv, read_run
from .fit import predict_probabilities, use_full_precision
from .images import load_spectrograms
from .saved import load_model, run_problems
```

and replace the start of `verify_run`, from `metadata = read_run(run_dir)` through `model.to(device)`, with:

```python
    metadata = read_run(run_dir)
    problems = run_problems(run_dir, metadata)
    if problems:
        return VerificationResult(0, 0.0, tuple(problems))

    use_full_precision()
    model = load_model(run_dir).to(device)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.training.test_saved tests.training.test_verify tests.training.test_run tests.training.test_commands tests.test_repository_reproducibility -v`
Expected: every test passes.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/training/saved.py src/frog_classifier/training/artifact.py src/frog_classifier/training/run.py src/frog_classifier/training/verify.py tests/training/test_saved.py tests/test_repository_reproducibility.py
git commit -m "feat: load a checked saved run for prediction"
```

---

### Task 3: Finding recordings and reading their start times

**Files:**
- Create: `src/frog_classifier/inference/__init__.py`
- Create: `src/frog_classifier/inference/recordings.py`
- Create: `tests/inference/__init__.py` (empty)
- Create: `tests/inference/helpers.py`
- Test: `tests/inference/test_recordings.py`

**Interfaces:**
- Produces: `AUDIO_SUFFIXES`, `NO_START_TIME`, `RecordingTime(started_at: datetime | None, utc_offset: str | None, notes: tuple[str, ...])`, `find_recordings(input_root: Path) -> list[str]`, `recording_time(path: Path) -> RecordingTime`, `audiomoth_comment(path: Path) -> str | None`.
- Produces (tests): `tests.inference.helpers.AUDIOMOTH_COMMENT`, `riff_chunk(chunk_id: bytes, data: bytes) -> bytes`, `write_recording(path, waveform, *, sample_rate_hz=48000, comment=None, extra_chunks=b"") -> Path`.

- [ ] **Step 1: Write the failing tests**

Create `tests/inference/__init__.py` as an empty file.

Create `tests/inference/helpers.py`:

```python
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


AUDIOMOTH_COMMENT = (
    "Recorded at 01:00:00 23/01/2022 (UTC+11) by AudioMoth 248D9B0260370525 "
    "at medium gain while battery was 4.0V and temperature was 20.4C."
)


def riff_chunk(chunk_id: bytes, data: bytes) -> bytes:
    """One RIFF chunk, padded to an even length as the format requires."""
    return chunk_id + struct.pack("<I", len(data)) + data + b"\0" * (len(data) % 2)


def write_recording(
    path: Path,
    waveform: np.ndarray,
    *,
    sample_rate_hz: int = 48000,
    comment: str | None = None,
    extra_chunks: bytes = b"",
) -> Path:
    """A 16-bit mono WAV laid out like an AudioMoth file: the format chunk, any
    extra chunks, an optional LIST INFO comment, then the samples."""
    samples = (np.clip(waveform, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate_hz, sample_rate_hz * 2, 2, 16)
    info = b""
    if comment is not None:
        info = riff_chunk(b"LIST", b"INFO" + riff_chunk(b"ICMT", comment.encode("ascii") + b"\0"))
    body = b"WAVE" + riff_chunk(b"fmt ", fmt) + extra_chunks + info + riff_chunk(b"data", samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    return path
```

Create `tests/inference/test_recordings.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from frog_classifier.inference.recordings import (
    NO_START_TIME,
    audiomoth_comment,
    find_recordings,
    recording_time,
)
from tests.inference.helpers import AUDIOMOTH_COMMENT, riff_chunk, write_recording
from tests.preprocessing.helpers import noise


class FindRecordingsTests(unittest.TestCase):
    def test_finds_audio_files_in_every_subfolder_in_sorted_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "b/20220102_000000.WAV",
                "a/x.mp3",
                "a/deeper/y.wav",
                "a/notes.txt",
                "a/CONFIG.TXT",
                "c d/\u00e9t\u00e9 1.Wav",
            ):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_bytes(b"")

            self.assertEqual(
                find_recordings(root),
                ["a/deeper/y.wav", "a/x.mp3", "b/20220102_000000.WAV", "c d/\u00e9t\u00e9 1.Wav"],
            )


class RecordingTimeTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _recording(self, name: str, comment: str | None = None, *, extra_chunks: bytes = b"") -> Path:
        return write_recording(
            self.root / name,
            noise(1, sample_rate_hz=48000),
            comment=comment,
            extra_chunks=extra_chunks,
        )

    def test_the_file_name_gives_the_start_time(self) -> None:
        time = recording_time(self._recording("20220123_010000.WAV"))

        self.assertEqual(time.started_at, datetime(2022, 1, 23, 1, 0, 0))
        self.assertIsNone(time.utc_offset)
        self.assertEqual(time.notes, ())

    def test_the_audiomoth_header_gives_the_utc_offset(self) -> None:
        time = recording_time(self._recording("20220123_010000.WAV", AUDIOMOTH_COMMENT))

        self.assertEqual(time.started_at, datetime(2022, 1, 23, 1, 0, 0))
        self.assertEqual(time.utc_offset, "+11:00")
        self.assertEqual(time.notes, ())

    def test_offsets_are_signed_hours_and_minutes(self) -> None:
        cases = (("UTC", "+00:00"), ("UTC-3", "-03:00"), ("UTC+9:30", "+09:30"), ("UTC+10", "+10:00"))
        for index, (zone, expected) in enumerate(cases):
            with self.subTest(zone=zone):
                comment = f"Recorded at 01:00:00 23/01/2022 ({zone}) by AudioMoth 248D9B0260370525."
                path = self._recording(f"zone{index}/20220123_010000.WAV", comment)

                self.assertEqual(recording_time(path).utc_offset, expected)

    def test_the_header_gives_the_start_time_when_the_file_name_has_none(self) -> None:
        time = recording_time(self._recording("site-a.wav", AUDIOMOTH_COMMENT))

        self.assertEqual(time.started_at, datetime(2022, 1, 23, 1, 0, 0))
        self.assertEqual(time.notes, ("start time taken from the AudioMoth header",))

    def test_a_disagreeing_header_is_noted_and_the_file_name_wins(self) -> None:
        time = recording_time(self._recording("20220123_020000.WAV", AUDIOMOTH_COMMENT))

        self.assertEqual(time.started_at, datetime(2022, 1, 23, 2, 0, 0))
        self.assertEqual(time.notes, (
            "file name time 2022-01-23T02:00:00 differs from the AudioMoth header time "
            "2022-01-23T01:00:00; using the file name",
        ))

    def test_an_impossible_file_name_date_is_noted(self) -> None:
        time = recording_time(self._recording("20221340_010000.wav"))

        self.assertIsNone(time.started_at)
        self.assertEqual(time.notes, (
            "file name timestamp 20221340_010000 is not a valid date",
            NO_START_TIME,
        ))

    def test_a_recording_without_any_time_says_why(self) -> None:
        time = recording_time(self._recording("site-a.wav"))

        self.assertIsNone(time.started_at)
        self.assertIsNone(time.utc_offset)
        self.assertEqual(time.notes, (NO_START_TIME,))

    def test_the_comment_is_found_after_other_chunks_of_odd_length(self) -> None:
        other_chunks = riff_chunk(b"junk", b"abc") + riff_chunk(b"LIST", b"adtl" + riff_chunk(b"note", b"x"))
        path = self._recording("20220123_010000.WAV", AUDIOMOTH_COMMENT, extra_chunks=other_chunks)

        self.assertEqual(audiomoth_comment(path), AUDIOMOTH_COMMENT)

    def test_a_file_that_is_not_a_wav_has_no_comment(self) -> None:
        path = self.root / "20220123_010000.mp3"
        path.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00")

        self.assertIsNone(audiomoth_comment(path))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.inference.test_recordings -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'frog_classifier.inference'`.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/inference/__init__.py`:

```python
"""Batch detection: run a saved model over whole folders of recordings.

Nothing is re-exported here: the detection module imports PyTorch, while the
modules that rendering workers import must stay free of it. Import modules
directly.
"""
```

Create `src/frog_classifier/inference/recordings.py`:

```python
"""Find recordings beneath an input folder and read when each one started."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


AUDIO_SUFFIXES = frozenset({".wav", ".mp3"})
NO_START_TIME = (
    "no start time: the file name has no valid YYYYMMDD_HHMMSS timestamp "
    "and there is no AudioMoth header"
)

_FILE_NAME_TIMESTAMP = re.compile(r"(?<!\d)\d{8}_\d{6}(?!\d)")
# AudioMoth firmware writes, for example,
# "Recorded at 01:00:00 23/01/2022 (UTC+11) by AudioMoth 248D9B0260370525 ...".
_AUDIOMOTH_TIME = re.compile(
    r"Recorded at (?P<clock>\d{2}:\d{2}:\d{2} \d{2}/\d{2}/\d{4}) "
    r"\(UTC(?:(?P<sign>[+-])(?P<hours>\d{1,2})(?::(?P<minutes>\d{2}))?)?\)"
)
# An INFO list holds a few short text fields; anything larger is not worth reading.
_MAX_LIST_BYTES = 1 << 20


@dataclass(frozen=True)
class RecordingTime:
    started_at: datetime | None
    utc_offset: str | None
    notes: tuple[str, ...]


def find_recordings(input_root: Path) -> list[str]:
    """Every audio file beneath input_root, as sorted POSIX paths relative to it."""
    return sorted(
        path.relative_to(input_root).as_posix()
        for path in input_root.rglob("*")
        if path.suffix.lower() in AUDIO_SUFFIXES and path.is_file()
    )


def recording_time(path: Path) -> RecordingTime:
    """The local start time of a recording and its UTC offset when known.

    The file name's YYYYMMDD_HHMMSS timestamp is the start time, as everywhere
    else in the project. An AudioMoth header supplies the UTC offset, and the
    start time when the file name has none.
    """
    named, name_note = _file_name_time(path.stem)
    notes = [name_note] if name_note is not None else []
    header_time, utc_offset = _audiomoth_time(path)
    if named is not None and header_time is not None and named != header_time:
        notes.append(
            f"file name time {named.isoformat()} differs from the AudioMoth header time "
            f"{header_time.isoformat()}; using the file name"
        )
    elif named is None and header_time is not None:
        notes.append("start time taken from the AudioMoth header")
    elif named is None:
        notes.append(NO_START_TIME)
    started_at = named if named is not None else header_time
    return RecordingTime(started_at, utc_offset, tuple(notes))


def audiomoth_comment(path: Path) -> str | None:
    """The INFO comment of a WAV file, where AudioMoth records its clock and time zone."""
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            return None
        while len(chunk_header := handle.read(8)) == 8:
            chunk_id = chunk_header[:4]
            (size,) = struct.unpack("<I", chunk_header[4:])
            padded = size + size % 2
            if chunk_id == b"LIST" and size <= _MAX_LIST_BYTES:
                body = handle.read(padded)[:size]
                comment = _info_comment(body) if body[:4] == b"INFO" else None
                if comment is not None:
                    return comment
            else:
                handle.seek(padded, 1)
    return None


def _info_comment(info: bytes) -> str | None:
    offset = 4
    while offset + 8 <= len(info):
        field_id = info[offset : offset + 4]
        (size,) = struct.unpack("<I", info[offset + 4 : offset + 8])
        if field_id == b"ICMT":
            return info[offset + 8 : offset + 8 + size].split(b"\0", 1)[0].decode("latin-1")
        offset += 8 + size + size % 2
    return None


def _file_name_time(stem: str) -> tuple[datetime | None, str | None]:
    match = _FILE_NAME_TIMESTAMP.search(stem)
    if match is None:
        return None, None
    try:
        return datetime.strptime(match.group(), "%Y%m%d_%H%M%S"), None
    except ValueError:
        return None, f"file name timestamp {match.group()} is not a valid date"


def _audiomoth_time(path: Path) -> tuple[datetime | None, str | None]:
    comment = audiomoth_comment(path)
    match = _AUDIOMOTH_TIME.search(comment) if comment is not None else None
    if match is None:
        return None, None
    try:
        started_at = datetime.strptime(match["clock"], "%H:%M:%S %d/%m/%Y")
    except ValueError:
        return None, None
    hours, minutes = int(match["hours"] or 0), int(match["minutes"] or 0)
    return started_at, f"{match['sign'] or '+'}{hours:02d}:{minutes:02d}"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.inference.test_recordings -v`
Expected: all ten tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/inference/__init__.py src/frog_classifier/inference/recordings.py tests/inference/__init__.py tests/inference/helpers.py tests/inference/test_recordings.py
git commit -m "feat: find recordings and read their start time and UTC offset"
```

---

### Task 4: Rendering one recording in a worker

**Files:**
- Create: `src/frog_classifier/inference/rendering.py`
- Test: `tests/inference/test_rendering.py`

**Interfaces:**
- Consumes: `load_waveform`, `render_waveform_images` (Task 1); `recording_time`, `RecordingTime` (Task 3); `expected_image_shape(config) -> tuple[int, int]` from `frog_classifier.training.images`.
- Produces: `RenderJob(input_root: Path, path: str, config: PreprocessingConfig)`; `RenderedRecording(path: str, recording_id: str, started_at: datetime | None, utc_offset: str | None, size_bytes: int | None, audio_sha256: str | None, duration_s: float | None, window_starts: tuple[int, ...], images: np.ndarray, notes: tuple[str, ...], error: str | None, render_seconds: float)`, where `images` is `uint8` with shape `(len(window_starts), n_mels, frames)`; `render_for_detection(job: RenderJob) -> RenderedRecording`; `start_render_worker() -> None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/inference/test_rendering.py`:

```python
from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from frog_classifier.inference.rendering import RenderJob, render_for_detection
from frog_classifier.preprocessing import load_waveform, render_waveform_images
from tests.data.helpers import load_test_config
from tests.inference.helpers import AUDIOMOTH_COMMENT, write_recording
from tests.preprocessing.helpers import noise


class RenderForDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = load_test_config(self.root)
        self.input_root = self.root / "raw"

    def _render(self, path: str):
        return render_for_detection(RenderJob(self.input_root, path, self.config))

    def test_renders_every_window_with_the_slicer_pixels_and_describes_the_source(self) -> None:
        source = write_recording(
            self.input_root / "site" / "20220123_010000.WAV",
            noise(12, sample_rate_hz=48000),
            comment=AUDIOMOTH_COMMENT,
        )

        rendered = self._render("site/20220123_010000.WAV")

        expected = np.stack([
            image.pixels
            for image in render_waveform_images(load_waveform(source, self.config), self.config)
        ])
        self.assertIsNone(rendered.error)
        self.assertEqual(rendered.path, "site/20220123_010000.WAV")
        self.assertEqual(rendered.recording_id, "20220123_010000")
        self.assertEqual(rendered.window_starts, (0, 5))
        np.testing.assert_array_equal(rendered.images, expected)
        self.assertEqual(rendered.started_at, datetime(2022, 1, 23, 1, 0, 0))
        self.assertEqual(rendered.utc_offset, "+11:00")
        self.assertEqual(rendered.size_bytes, source.stat().st_size)
        self.assertEqual(rendered.audio_sha256, hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertAlmostEqual(rendered.duration_s, 12.0, places=3)
        self.assertEqual(rendered.notes, ())
        self.assertGreater(rendered.render_seconds, 0.0)

    def test_a_recording_shorter_than_one_window_has_no_windows_and_says_so(self) -> None:
        write_recording(self.input_root / "20220123_010000.wav", noise(3, sample_rate_hz=48000))

        rendered = self._render("20220123_010000.wav")

        self.assertIsNone(rendered.error)
        self.assertEqual(rendered.window_starts, ())
        self.assertEqual(rendered.images.shape, (0, 128, 216))
        self.assertEqual(rendered.notes, ("shorter than one 5-second window",))

    def test_an_unreadable_file_is_a_failure_not_an_exception(self) -> None:
        path = self.input_root / "20220123_010000.wav"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not audio")

        with warnings.catch_warnings():
            # librosa warns before trying its fallback decoder; the failure is what is under test.
            warnings.simplefilter("ignore")
            rendered = self._render("20220123_010000.wav")

        self.assertIsNotNone(rendered.error)
        self.assertEqual(rendered.window_starts, ())
        self.assertEqual(rendered.images.shape, (0, 128, 216))
        self.assertEqual(rendered.size_bytes, 9)
        self.assertEqual(rendered.started_at, datetime(2022, 1, 23, 1, 0, 0))

    def test_a_file_that_vanished_is_a_failure_with_no_size(self) -> None:
        rendered = self._render("gone/20220123_010000.wav")

        self.assertIn("FileNotFoundError", rendered.error)
        self.assertIsNone(rendered.size_bytes)
        self.assertIsNone(rendered.started_at)

    def test_rendering_workers_do_not_load_pytorch(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import sys, frog_classifier.inference.rendering; print('torch' in sys.modules)"],
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.inference.test_rendering -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'frog_classifier.inference.rendering'`.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/inference/rendering.py`:

```python
"""Render one recording into window images for detection.

Rendering runs in worker processes, so this module and everything it imports
stay free of PyTorch.
"""

from __future__ import annotations

import hashlib
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np

from frog_classifier.data.config import PreprocessingConfig
from frog_classifier.preprocessing import ChunkImage, load_waveform, render_waveform_images
from frog_classifier.training.images import expected_image_shape

from .recordings import RecordingTime, recording_time


_HASH_BLOCK_BYTES = 1 << 20


@dataclass(frozen=True)
class RenderJob:
    input_root: Path
    path: str
    config: PreprocessingConfig


@dataclass(frozen=True)
class RenderedRecording:
    path: str
    recording_id: str
    started_at: datetime | None
    utc_offset: str | None
    size_bytes: int | None
    audio_sha256: str | None
    duration_s: float | None
    window_starts: tuple[int, ...]
    images: np.ndarray
    notes: tuple[str, ...]
    error: str | None
    render_seconds: float


def start_render_worker() -> None:
    """Leave Ctrl+C to the main process, which cancels the remaining work and
    stops the workers itself."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def render_for_detection(job: RenderJob) -> RenderedRecording:
    """Read, hash, time, and render one recording.

    Any error while doing so becomes this recording's failure, so one bad file
    never stops a batch.
    """
    started = time.perf_counter()
    source = job.input_root / job.path
    size_bytes: int | None = None
    audio_sha256: str | None = None
    duration_s: float | None = None
    error: str | None = None
    timing = RecordingTime(None, None, ())
    chunks: list[ChunkImage] = []
    try:
        size_bytes = source.stat().st_size
        audio_sha256 = _file_sha256(source)
        timing = recording_time(source)
        waveform = load_waveform(source, job.config)
        duration_s = len(waveform) / job.config.audio.sample_rate_hz
        chunks = list(render_waveform_images(waveform, job.config))
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    notes = timing.notes
    if error is None and not chunks:
        notes += (f"shorter than one {job.config.audio.chunk_seconds}-second window",)
    if chunks:
        images = np.stack([chunk.pixels for chunk in chunks])
    else:
        images = np.empty((0, *expected_image_shape(job.config)), dtype=np.uint8)
    return RenderedRecording(
        path=job.path,
        recording_id=PurePosixPath(job.path).stem,
        started_at=timing.started_at,
        utc_offset=timing.utc_offset,
        size_bytes=size_bytes,
        audio_sha256=audio_sha256,
        duration_s=duration_s,
        window_starts=tuple(chunk.start_s for chunk in chunks),
        images=images,
        notes=notes,
        error=error,
        render_seconds=time.perf_counter() - started,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.inference.test_rendering -v`
Expected: all five tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/inference/rendering.py tests/inference/test_rendering.py
git commit -m "feat: render one recording for detection without PyTorch"
```

---

### Task 5: The SQLite detection store

**Files:**
- Create: `src/frog_classifier/inference/store.py`
- Modify: `tests/inference/helpers.py`
- Test: `tests/inference/test_store.py`

**Interfaces:**
- Produces: `SCHEMA_VERSION = 1`, `DATABASE_FILE = "detections.sqlite"`, `StoreError(ValueError)`, `ScanIdentity(input_root: str, model_run: str, model_sha256: str, threshold: float, preprocessing_sha256: str, chunk_seconds: int)`, `RecordingRow(path, recording_id, started_at, utc_offset, size_bytes, audio_sha256, duration_s, windows, status, message, render_seconds, predict_seconds, processed_at)` with the column types of the spec's `recordings` table, and `DetectionStore` with `DetectionStore.open(path: Path, identity: ScanIdentity, *, now: str) -> DetectionStore`, attributes `identity` and `created_at`, `finished_paths() -> set[str]`, `save(recording: RecordingRow, window_starts: Sequence[int], probabilities: Sequence[float]) -> None`, `recording_rows() -> list[RecordingRow]` ordered by path, `windows(*, minimum_probability: float = 0.0) -> list[tuple[str, int, float]]` ordered by path then start, `close()`, and use as a context manager.
- Produces (tests): `tests.inference.helpers.SCAN_IDENTITY` and `recording_row(path: str, **changes) -> RecordingRow`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/inference/helpers.py` (add `from dataclasses import replace` and `from pathlib import Path, PurePosixPath` to its imports, and `from frog_classifier.inference.store import RecordingRow, ScanIdentity`):

```python
SCAN_IDENTITY = ScanIdentity(
    input_root="D:/recordings/site-a",
    model_run="resnet18-20261014T114432Z",
    model_sha256="a" * 64,
    threshold=0.5,
    preprocessing_sha256="b" * 64,
    chunk_seconds=5,
)


def recording_row(path: str, **changes: object) -> RecordingRow:
    """A successfully processed two-window recording, with any fields replaced."""
    row = RecordingRow(
        path=path,
        recording_id=PurePosixPath(path).stem,
        started_at="2022-01-23T01:00:00",
        utc_offset="+11:00",
        size_bytes=1000,
        audio_sha256="c" * 64,
        duration_s=10.0,
        windows=2,
        status="ok",
        message=None,
        render_seconds=0.5,
        predict_seconds=0.01,
        processed_at="2026-10-15T00:00:00Z",
    )
    return replace(row, **changes)
```

Create `tests/inference/test_store.py`:

```python
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from frog_classifier.inference.store import DATABASE_FILE, DetectionStore, ScanIdentity, StoreError
from tests.inference.helpers import SCAN_IDENTITY, recording_row


class DetectionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / DATABASE_FILE

    def _open(self, identity: ScanIdentity = SCAN_IDENTITY, now: str = "2026-10-15T00:00:00Z") -> DetectionStore:
        store = DetectionStore.open(self.path, identity, now=now)
        self.addCleanup(store.close)
        return store

    def _execute(self, statement: str) -> None:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute(statement)
            connection.commit()
        finally:
            connection.close()

    def test_a_reopened_database_keeps_its_identity_and_creation_time(self) -> None:
        self._open().close()

        reopened = self._open(now="2026-10-16T00:00:00Z")

        self.assertEqual(reopened.identity, SCAN_IDENTITY)
        self.assertEqual(reopened.created_at, "2026-10-15T00:00:00Z")

    def test_a_database_made_for_another_scan_is_refused(self) -> None:
        self._open().close()
        changes = {
            "input_root": "E:/elsewhere",
            "model_sha256": "e" * 64,
            "threshold": 0.75,
            "preprocessing_sha256": "f" * 64,
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(StoreError, f"different {field}; choose another output folder"):
                    DetectionStore.open(self.path, replace(SCAN_IDENTITY, **{field: value}), now="later")

    def test_a_database_from_another_schema_version_is_refused(self) -> None:
        self._open().close()
        self._execute("UPDATE scan SET schema_version = 2")

        with self.assertRaisesRegex(StoreError, "schema 2"):
            DetectionStore.open(self.path, SCAN_IDENTITY, now="later")

    def test_a_file_that_is_not_a_database_is_refused(self) -> None:
        self.path.write_bytes(b"these bytes are not a database " * 40)

        with self.assertRaisesRegex(StoreError, "is not a detection database"):
            DetectionStore.open(self.path, SCAN_IDENTITY, now="later")

    def test_another_sqlite_database_is_refused(self) -> None:
        self._execute("CREATE TABLE other (value INTEGER)")

        with self.assertRaisesRegex(StoreError, "is not a detection database"):
            DetectionStore.open(self.path, SCAN_IDENTITY, now="later")

    def test_a_recording_is_stored_with_all_of_its_windows(self) -> None:
        store = self._open()
        store.save(recording_row("site/a.wav"), (0, 5), (0.25, 0.75))
        store.close()

        reopened = self._open()

        self.assertEqual(reopened.recording_rows(), [recording_row("site/a.wav")])
        self.assertEqual(reopened.windows(), [("site/a.wav", 0, 0.25), ("site/a.wav", 5, 0.75)])
        self.assertEqual(reopened.finished_paths(), {"site/a.wav"})

    def test_a_save_that_fails_partway_stores_nothing(self) -> None:
        store = self._open()

        with self.assertRaises(sqlite3.IntegrityError):
            store.save(recording_row("site/a.wav"), (0, 0), (0.25, 0.75))

        self.assertEqual(store.recording_rows(), [])
        self.assertEqual(store.windows(), [])

    def test_a_retried_recording_replaces_its_failed_attempt(self) -> None:
        store = self._open()
        store.save(recording_row("site/a.wav", windows=0, status="failed", message="OSError: busy"), (), ())
        self.assertEqual(store.finished_paths(), set())

        store.save(recording_row("site/a.wav"), (0, 5), (0.25, 0.75))

        self.assertEqual(store.recording_rows(), [recording_row("site/a.wav")])
        self.assertEqual(len(store.windows()), 2)

    def test_windows_can_be_limited_to_a_minimum_probability(self) -> None:
        store = self._open()
        store.save(recording_row("site/b.wav", windows=3), (0, 5, 10), (0.2, 0.5, 0.9))
        store.save(recording_row("site/a.wav", windows=1), (0,), (0.6,))

        self.assertEqual(
            store.windows(minimum_probability=0.5),
            [("site/a.wav", 0, 0.6), ("site/b.wav", 5, 0.5), ("site/b.wav", 10, 0.9)],
        )

    def test_a_window_count_that_disagrees_with_the_windows_is_rejected(self) -> None:
        store = self._open()

        with self.assertRaises(ValueError):
            store.save(recording_row("site/a.wav", windows=3), (0, 5), (0.25, 0.75))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.inference.test_store -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'frog_classifier.inference.store'`.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/inference/store.py`:

```python
"""The detection record: one SQLite database per output folder.

Each recording is stored in a single transaction together with all of its
windows, so an interrupted run leaves only whole recordings behind and the
next run continues from them.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Iterator, Sequence


SCHEMA_VERSION = 1
DATABASE_FILE = "detections.sqlite"

_TABLES = (
    """CREATE TABLE scan (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        schema_version INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        input_root TEXT NOT NULL,
        model_run TEXT NOT NULL,
        model_sha256 TEXT NOT NULL,
        threshold REAL NOT NULL,
        preprocessing_sha256 TEXT NOT NULL,
        chunk_seconds INTEGER NOT NULL
    )""",
    """CREATE TABLE recordings (
        path TEXT PRIMARY KEY,
        recording_id TEXT NOT NULL,
        started_at TEXT,
        utc_offset TEXT,
        size_bytes INTEGER,
        audio_sha256 TEXT,
        duration_s REAL,
        windows INTEGER NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('ok', 'failed')),
        message TEXT,
        render_seconds REAL NOT NULL,
        predict_seconds REAL NOT NULL,
        processed_at TEXT NOT NULL
    )""",
    """CREATE TABLE windows (
        path TEXT NOT NULL REFERENCES recordings (path),
        start_s INTEGER NOT NULL,
        probability REAL NOT NULL,
        PRIMARY KEY (path, start_s)
    )""",
)


class StoreError(ValueError):
    pass


@dataclass(frozen=True)
class ScanIdentity:
    """What a database's results depend on; one database never mixes two of these."""

    input_root: str
    model_run: str
    model_sha256: str
    threshold: float
    preprocessing_sha256: str
    chunk_seconds: int


@dataclass(frozen=True)
class RecordingRow:
    path: str
    recording_id: str
    started_at: str | None
    utc_offset: str | None
    size_bytes: int | None
    audio_sha256: str | None
    duration_s: float | None
    windows: int
    status: str
    message: str | None
    render_seconds: float
    predict_seconds: float
    processed_at: str


_IDENTITY_COLUMNS = tuple(field.name for field in fields(ScanIdentity))
_RECORDING_COLUMNS = tuple(field.name for field in fields(RecordingRow))


class DetectionStore:
    def __init__(self, connection: sqlite3.Connection, identity: ScanIdentity, created_at: str) -> None:
        self._connection = connection
        self.identity = identity
        self.created_at = created_at

    @classmethod
    def open(cls, path: Path, identity: ScanIdentity, *, now: str) -> DetectionStore:
        """Create the database for a new scan, or reopen one made for exactly this scan."""
        # Transactions are explicit: autocommit mode, with BEGIN and COMMIT issued here.
        connection = sqlite3.connect(path, isolation_level=None)
        try:
            tables = _table_names(connection, path)
            connection.execute("PRAGMA foreign_keys = ON")
            if not tables:
                created_at = _create(connection, identity, now)
            elif "scan" not in tables:
                raise StoreError(f"{path} is not a detection database; choose another output folder")
            else:
                created_at = _check_scan(connection, path, identity)
        except BaseException:
            connection.close()
            raise
        return cls(connection, identity, created_at)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> DetectionStore:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def finished_paths(self) -> set[str]:
        """Recordings stored successfully; a failed recording is not finished and is retried."""
        rows = self._connection.execute("SELECT path FROM recordings WHERE status = 'ok'")
        return {path for (path,) in rows}

    def save(self, recording: RecordingRow, window_starts: Sequence[int], probabilities: Sequence[float]) -> None:
        """Store one recording with all of its windows, replacing any earlier attempt."""
        if not recording.windows == len(window_starts) == len(probabilities):
            raise ValueError("a recording's window count must match its window starts and probabilities")
        columns = ", ".join(_RECORDING_COLUMNS)
        placeholders = ", ".join("?" * len(_RECORDING_COLUMNS))
        with _transaction(self._connection):
            self._connection.execute("DELETE FROM windows WHERE path = ?", (recording.path,))
            self._connection.execute("DELETE FROM recordings WHERE path = ?", (recording.path,))
            self._connection.execute(
                f"INSERT INTO recordings ({columns}) VALUES ({placeholders})",
                astuple(recording),
            )
            self._connection.executemany(
                "INSERT INTO windows (path, start_s, probability) VALUES (?, ?, ?)",
                [
                    (recording.path, int(start), float(probability))
                    for start, probability in zip(window_starts, probabilities, strict=True)
                ],
            )

    def recording_rows(self) -> list[RecordingRow]:
        query = f"SELECT {', '.join(_RECORDING_COLUMNS)} FROM recordings ORDER BY path"
        return [RecordingRow(*row) for row in self._connection.execute(query)]

    def windows(self, *, minimum_probability: float = 0.0) -> list[tuple[str, int, float]]:
        """(path, start_s, probability) of every window at or above minimum_probability."""
        return list(self._connection.execute(
            "SELECT path, start_s, probability FROM windows WHERE probability >= ? ORDER BY path, start_s",
            (minimum_probability,),
        ))


@contextmanager
def _transaction(connection: sqlite3.Connection) -> Iterator[None]:
    connection.execute("BEGIN")
    try:
        yield
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")


def _table_names(connection: sqlite3.Connection, path: Path) -> set[str]:
    try:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    except sqlite3.DatabaseError as error:
        raise StoreError(f"{path} is not a detection database ({error}); choose another output folder") from error
    return {name for (name,) in rows}


def _create(connection: sqlite3.Connection, identity: ScanIdentity, now: str) -> str:
    columns = ", ".join(_IDENTITY_COLUMNS)
    placeholders = ", ".join("?" * len(_IDENTITY_COLUMNS))
    with _transaction(connection):
        for statement in _TABLES:
            connection.execute(statement)
        connection.execute(
            f"INSERT INTO scan (id, schema_version, created_at, {columns}) VALUES (1, ?, ?, {placeholders})",
            (SCHEMA_VERSION, now, *astuple(identity)),
        )
    return now


def _check_scan(connection: sqlite3.Connection, path: Path, identity: ScanIdentity) -> str:
    (version,) = connection.execute("SELECT schema_version FROM scan").fetchone()
    if version != SCHEMA_VERSION:
        raise StoreError(
            f"{path} uses detection schema {version}, but this code reads schema "
            f"{SCHEMA_VERSION}; choose another output folder"
        )
    row = connection.execute(f"SELECT created_at, {', '.join(_IDENTITY_COLUMNS)} FROM scan").fetchone()
    stored = ScanIdentity(*row[1:])
    changed = [name for name in _IDENTITY_COLUMNS if getattr(stored, name) != getattr(identity, name)]
    if changed:
        raise StoreError(
            f"{path.parent} holds results for a different {', '.join(changed)}; choose another output folder"
        )
    return row[0]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.inference.test_store tests.inference.test_recordings -v`
Expected: every test passes.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/inference/store.py tests/inference/helpers.py tests/inference/test_store.py
git commit -m "feat: store detections in a resumable SQLite record"
```

---

### Task 6: Spreadsheet export

**Files:**
- Create: `src/frog_classifier/inference/export.py`
- Test: `tests/inference/test_export.py`

**Interfaces:**
- Consumes: `DetectionStore` (Task 5) and `write_csv(path, columns, rows)` from `frog_classifier.training.artifact`.
- Produces: `DETECTIONS_CSV`, `RECORDINGS_CSV`, `DETECTION_COLUMNS`, `RECORDING_COLUMNS`, `ExportError(Exception)`, `export_csv(store: DetectionStore, output_dir: Path) -> None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/inference/test_export.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frog_classifier.inference.export import (
    DETECTIONS_CSV,
    RECORDINGS_CSV,
    ExportError,
    export_csv,
)
from frog_classifier.inference.store import DATABASE_FILE, DetectionStore
from frog_classifier.training.artifact import read_csv
from tests.inference.helpers import SCAN_IDENTITY, recording_row


FAILED_PATH = "site/\u00e9t\u00e9 2.wav"


class ExportTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.store = DetectionStore.open(self.output / DATABASE_FILE, SCAN_IDENTITY, now="2026-10-15T00:00:00Z")
        self.addCleanup(self.store.close)
        self.store.save(recording_row("site/20220123_010000.WAV", windows=3), (0, 5, 10), (0.2, 0.5, 0.9))
        self.store.save(
            recording_row(
                FAILED_PATH, windows=0, status="failed", message="LibsndfileError: bad header",
                started_at=None, utc_offset=None, duration_s=None, audio_sha256=None,
            ),
            (),
            (),
        )

    def test_detections_are_the_windows_at_or_above_the_threshold_with_their_provenance(self) -> None:
        export_csv(self.store, self.output)

        rows = read_csv(self.output / DETECTIONS_CSV)
        self.assertEqual([row["window_start_s"] for row in rows], ["5", "10"])
        self.assertEqual(rows[0], {
            "recording_path": "site/20220123_010000.WAV",
            "recording_id": "20220123_010000",
            "window_start_s": "5",
            "window_end_s": "10",
            "window_start_time": "2022-01-23T01:00:05",
            "utc_offset": "+11:00",
            "probability": "0.50000000",
            "threshold": "0.50000000",
            "model_run": SCAN_IDENTITY.model_run,
            "model_sha256": SCAN_IDENTITY.model_sha256,
            "preprocessing_sha256": SCAN_IDENTITY.preprocessing_sha256,
        })

    def test_every_recording_is_listed_with_its_status_and_detection_count(self) -> None:
        export_csv(self.store, self.output)

        rows = read_csv(self.output / RECORDINGS_CSV)
        self.assertEqual(
            [(row["recording_path"], row["status"], row["windows"], row["detections"]) for row in rows],
            [("site/20220123_010000.WAV", "ok", "3", "2"), (FAILED_PATH, "failed", "0", "0")],
        )
        self.assertEqual(rows[0]["started_at"], "2022-01-23T01:00:00")
        self.assertEqual(rows[0]["duration_s"], "10.000")
        self.assertEqual(rows[1]["started_at"], "")
        self.assertEqual(rows[1]["message"], "LibsndfileError: bad header")

    def test_a_csv_that_cannot_be_replaced_is_explained_and_leaves_no_partial_file(self) -> None:
        with patch("frog_classifier.inference.export.os.replace", side_effect=PermissionError(13, "Permission denied")):
            with self.assertRaisesRegex(ExportError, "detections.csv could not be replaced.*run the same command again"):
                export_csv(self.store, self.output)

        self.assertEqual(sorted(path.name for path in self.output.iterdir()), [DATABASE_FILE])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.inference.test_export -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'frog_classifier.inference.export'`.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/inference/export.py`:

```python
"""Spreadsheet copies of the detection record, rewritten from the database."""

from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from frog_classifier.training.artifact import write_csv

from .store import DetectionStore


DETECTIONS_CSV = "detections.csv"
RECORDINGS_CSV = "recordings.csv"
DETECTION_COLUMNS = (
    "recording_path", "recording_id", "window_start_s", "window_end_s", "window_start_time",
    "utc_offset", "probability", "threshold", "model_run", "model_sha256", "preprocessing_sha256",
)
RECORDING_COLUMNS = (
    "recording_path", "recording_id", "started_at", "utc_offset", "status", "windows",
    "detections", "duration_s", "audio_sha256", "message",
)


class ExportError(Exception):
    pass


def export_csv(store: DetectionStore, output_dir: Path) -> None:
    """Rewrite detections.csv and recordings.csv from the database.

    A detection is a window whose probability is at or above the run's threshold.
    """
    identity = store.identity
    recordings = {row.path: row for row in store.recording_rows()}
    detected = store.windows(minimum_probability=identity.threshold)
    counts = Counter(path for path, _, _ in detected)
    _replace(output_dir / DETECTIONS_CSV, DETECTION_COLUMNS, (
        {
            "recording_path": path,
            "recording_id": recordings[path].recording_id,
            "window_start_s": start_s,
            "window_end_s": start_s + identity.chunk_seconds,
            "window_start_time": _clock_time(recordings[path].started_at, start_s),
            "utc_offset": recordings[path].utc_offset,
            "probability": f"{probability:.8f}",
            "threshold": f"{identity.threshold:.8f}",
            "model_run": identity.model_run,
            "model_sha256": identity.model_sha256,
            "preprocessing_sha256": identity.preprocessing_sha256,
        }
        for path, start_s, probability in detected
    ))
    _replace(output_dir / RECORDINGS_CSV, RECORDING_COLUMNS, (
        {
            "recording_path": row.path,
            "recording_id": row.recording_id,
            "started_at": row.started_at,
            "utc_offset": row.utc_offset,
            "status": row.status,
            "windows": row.windows,
            "detections": counts[row.path],
            "duration_s": None if row.duration_s is None else f"{row.duration_s:.3f}",
            "audio_sha256": row.audio_sha256,
            "message": row.message,
        }
        for row in recordings.values()
    ))


def _clock_time(started_at: str | None, offset_s: int) -> str | None:
    if started_at is None:
        return None
    return (datetime.fromisoformat(started_at) + timedelta(seconds=offset_s)).isoformat()


def _replace(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    """Write beside the target and then swap it in, so no reader sees half a file."""
    staging = path.with_name(f".{path.name}.partial")
    write_csv(staging, columns, rows)
    try:
        os.replace(staging, path)
    except PermissionError as error:
        staging.unlink()
        raise ExportError(
            f"{path.name} could not be replaced, probably because it is open in another program. "
            "The detection database is complete: close the file and run the same command again "
            "to rewrite the CSV files."
        ) from error
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.inference.test_export -v`
Expected: all three tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/inference/export.py tests/inference/test_export.py
git commit -m "feat: export detections and recordings as CSV copies"
```

---

### Task 7: The detection run

**Files:**
- Create: `src/frog_classifier/inference/detect.py`
- Modify: `tests/inference/helpers.py`
- Test: `tests/inference/test_detect.py`

**Interfaces:**
- Consumes: `load_saved_run`, `SavedRun` (Task 2); `find_recordings` (Task 3); `RenderJob`, `RenderedRecording`, `render_for_detection`, `start_render_worker` (Task 4); `DATABASE_FILE`, `DetectionStore`, `RecordingRow`, `ScanIdentity` (Task 5); `ExportError`, `export_csv` (Task 6); `predict_probabilities`, `use_full_precision` from `frog_classifier.training.fit`.
- Produces: `MAX_DEFAULT_WORKERS = 8`, `RECORDINGS_IN_FLIGHT_PER_WORKER = 2`, `DetectionError(ValueError)`, `RecordingOutcome(path: str, failed: bool, windows: int, detections: int, message: str | None)`, `DetectionSummary(output_dir: Path, threshold: float, found: int, already_done: int, processed: int, failed: int, windows: int, detections: int, audio_seconds: float, wall_seconds: float, render_seconds: float, predict_seconds: float, csv_problem: str | None)`, `default_workers() -> int`, `default_output_dir(results_root: Path, input_root: Path, run_name: str) -> Path`, `utc_now() -> datetime`, and `detect_calls(run_dir: Path, input_root: Path, output_dir: Path, *, device: torch.device, workers: int, on_start: Callable[[int, int], None] = ..., on_recording: Callable[[RecordingOutcome], None] = ..., clock: Callable[[], datetime] = utc_now) -> DetectionSummary`.
- Produces (tests): `tests.inference.helpers.build_saved_run(models_root: Path, *, name: str = "resnet18-20261015T000000Z", threshold: float = 0.5, seed: int = 0) -> Path`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/inference/helpers.py` (add `import torch`, `from tests.data.helpers import VALID_CONFIG`, and the training imports below to its imports):

```python
from frog_classifier.training.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    MODEL_FILE,
    PREPROCESSING_FILE,
    finish_run,
    start_run,
)
from frog_classifier.training.model import ARCHITECTURE, SpectrogramResNet
```

```python
def build_saved_run(
    models_root: Path,
    *,
    name: str = "resnet18-20261015T000000Z",
    threshold: float = 0.5,
    seed: int = 0,
) -> Path:
    """A saved run holding an untrained network: detection needs a valid run, not a good model."""
    torch.manual_seed(seed)
    staging = start_run(models_root, name)
    torch.save(SpectrogramResNet(pretrained=False).state_dict(), staging / MODEL_FILE)
    (staging / PREPROCESSING_FILE).write_bytes(VALID_CONFIG)
    return finish_run(staging, {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_name": name,
        "model": {"architecture": ARCHITECTURE, "input_shape": [1, 128, 216]},
        "threshold": {"value": threshold, "rule": "chosen by the test"},
    })
```

Create `tests/inference/test_detect.py`:

```python
from __future__ import annotations

import sqlite3
import tempfile
import unittest
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from frog_classifier.inference.detect import (
    DetectionError,
    DetectionSummary,
    RecordingOutcome,
    default_output_dir,
    detect_calls,
)
from frog_classifier.inference.export import DETECTIONS_CSV, RECORDINGS_CSV
from frog_classifier.inference.store import DATABASE_FILE, StoreError
from frog_classifier.preprocessing import decode_png, render_recording
from frog_classifier.training.fit import predict_probabilities
from frog_classifier.training.saved import load_saved_run
from tests.inference.helpers import AUDIOMOTH_COMMENT, build_saved_run, write_recording
from tests.preprocessing.helpers import noise


CPU = torch.device("cpu")


class Clock:
    """Each reading is one second after the last, so every stored time is distinct."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


class DetectCallsTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.run_dir = build_saved_run(self.root / "models", threshold=0.5)
        self.input_root = self.root / "raw" / "site"
        write_recording(
            self.input_root / "20220123_010000.WAV",
            noise(12, sample_rate_hz=48000, seed=1),
            comment=AUDIOMOTH_COMMENT,
        )
        write_recording(self.input_root / "20220123_020000.WAV", noise(10, sample_rate_hz=48000, seed=2))
        # The same file name in another folder is a different recording.
        write_recording(
            self.input_root / "nested" / "20220123_010000.WAV",
            noise(5, sample_rate_hz=22050, seed=3),
            sample_rate_hz=22050,
        )
        write_recording(self.input_root / "20220124_010000.wav", noise(3, sample_rate_hz=48000, seed=4))
        self.output = self.root / "results" / "detections" / "site"

    def _detect(self, output: Path | None = None, *, workers: int = 1, **options) -> DetectionSummary:
        return detect_calls(
            self.run_dir,
            self.input_root,
            output or self.output,
            device=CPU,
            workers=workers,
            **options,
        )

    def _query(self, statement: str, parameters: tuple = (), *, output: Path | None = None) -> list[tuple]:
        connection = sqlite3.connect((output or self.output) / DATABASE_FILE)
        try:
            return connection.execute(statement, parameters).fetchall()
        finally:
            connection.close()

    def test_every_window_of_every_recording_is_stored_and_exported(self) -> None:
        summary = self._detect()

        self.assertEqual((summary.found, summary.already_done, summary.processed, summary.failed), (4, 0, 4, 0))
        self.assertEqual(self._query("SELECT path, windows, status, message FROM recordings ORDER BY path"), [
            ("20220123_010000.WAV", 2, "ok", None),
            ("20220123_020000.WAV", 2, "ok", None),
            ("20220124_010000.wav", 0, "ok", "shorter than one 5-second window"),
            ("nested/20220123_010000.WAV", 1, "ok", None),
        ])
        self.assertEqual(self._query(
            "SELECT started_at, utc_offset FROM recordings WHERE path = ?", ("20220123_010000.WAV",),
        ), [("2022-01-23T01:00:00", "+11:00")])
        self.assertEqual(summary.windows, 5)
        self.assertEqual(
            summary.detections,
            self._query("SELECT COUNT(*) FROM windows WHERE probability >= 0.5")[0][0],
        )
        self.assertAlmostEqual(summary.audio_seconds, 30.0, places=3)
        self.assertIsNone(summary.csv_problem)
        self.assertTrue((self.output / DETECTIONS_CSV).is_file())
        self.assertTrue((self.output / RECORDINGS_CSV).is_file())

    def test_probabilities_equal_predictions_on_the_slicer_images(self) -> None:
        self._detect()

        saved = load_saved_run(self.run_dir)
        for relative in ("20220123_010000.WAV", "20220123_020000.WAV", "nested/20220123_010000.WAV"):
            with self.subTest(recording=relative):
                chunks = render_recording(self.input_root / relative, saved.preprocessing)
                pixels = np.stack([decode_png(chunk.png_bytes) for chunk in chunks])
                stored = self._query("SELECT probability FROM windows WHERE path = ? ORDER BY start_s", (relative,))

                np.testing.assert_allclose(
                    [probability for (probability,) in stored],
                    predict_probabilities(saved.model, pixels, CPU),
                    rtol=0,
                    atol=1e-9,
                )

    def test_an_unreadable_recording_is_recorded_as_failed_and_retried_next_time(self) -> None:
        (self.input_root / "20220125_010000.wav").write_bytes(b"not audio")

        with warnings.catch_warnings():
            # librosa warns before trying its fallback decoder; the recorded failure is what is under test.
            warnings.simplefilter("ignore")
            first = self._detect()
            second = self._detect()

        self.assertEqual((first.processed, first.failed), (5, 1))
        self.assertEqual((second.already_done, second.processed, second.failed), (4, 1, 1))
        ((status, message),) = self._query(
            "SELECT status, message FROM recordings WHERE path = ?", ("20220125_010000.wav",),
        )
        self.assertEqual(status, "failed")
        self.assertTrue(message)

    def test_an_interrupted_run_resumes_without_repeating_finished_recordings(self) -> None:
        for workers in (1, 2):
            with self.subTest(workers=workers):
                output = self.root / f"interrupted-{workers}"
                stored: list[str] = []

                def interrupt_after_two(outcome: RecordingOutcome) -> None:
                    stored.append(outcome.path)
                    if len(stored) == 2:
                        raise KeyboardInterrupt

                with self.assertRaises(KeyboardInterrupt):
                    self._detect(
                        output,
                        workers=workers,
                        on_recording=interrupt_after_two,
                        clock=Clock(datetime(2026, 10, 15, tzinfo=timezone.utc)),
                    )
                before = self._query("SELECT path, processed_at FROM recordings ORDER BY path", output=output)

                resumed = self._detect(output, workers=workers, clock=Clock(datetime(2026, 10, 16, tzinfo=timezone.utc)))

                self.assertEqual([path for path, _ in before], sorted(stored))
                self.assertEqual((resumed.already_done, resumed.processed), (2, 2))
                after = dict(self._query("SELECT path, processed_at FROM recordings", output=output))
                for path, processed_at in before:
                    self.assertEqual(after[path], processed_at)
                self.assertEqual(self._query("SELECT COUNT(*) FROM windows", output=output), [(5,)])

    def test_parallel_rendering_stores_the_same_record_as_one_worker(self) -> None:
        one, two = self.root / "one-worker", self.root / "two-workers"
        self._detect(one, workers=1)
        self._detect(two, workers=2)

        recordings = (
            "SELECT path, recording_id, started_at, utc_offset, size_bytes, audio_sha256, "
            "duration_s, windows, status, message FROM recordings ORDER BY path"
        )
        windows = "SELECT path, start_s, probability FROM windows ORDER BY path, start_s"
        self.assertEqual(self._query(recordings, output=one), self._query(recordings, output=two))
        self.assertEqual(self._query(windows, output=one), self._query(windows, output=two))

    def test_a_folder_with_results_from_another_model_is_refused(self) -> None:
        self._detect()
        other = build_saved_run(self.root / "models", name="resnet18-20261016T000000Z", seed=1)

        with self.assertRaisesRegex(StoreError, "different model_run, model_sha256"):
            detect_calls(other, self.input_root, self.output, device=CPU, workers=1)

    def test_an_output_folder_inside_the_input_folder_is_refused(self) -> None:
        with self.assertRaisesRegex(DetectionError, "must not lie inside the input folder"):
            self._detect(self.input_root / "results")

        self.assertFalse((self.input_root / "results").exists())

    def test_a_missing_input_folder_is_refused(self) -> None:
        with self.assertRaisesRegex(DetectionError, "input folder not found"):
            detect_calls(self.run_dir, self.root / "missing", self.output, device=CPU, workers=1)

    def test_the_default_output_folder_is_named_after_the_input_and_the_run(self) -> None:
        self.assertEqual(
            default_output_dir(self.root / "results", self.input_root, "resnet18-x"),
            self.root / "results" / "site-resnet18-x",
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.inference.test_detect -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'frog_classifier.inference.detect'`.

- [ ] **Step 3: Implement**

Create `src/frog_classifier/inference/detect.py`:

```python
"""One detection run: resume, render in parallel, predict, store, and export."""

from __future__ import annotations

import multiprocessing
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch

from frog_classifier.training.fit import predict_probabilities, use_full_precision
from frog_classifier.training.saved import SavedRun, load_saved_run

from .export import ExportError, export_csv
from .recordings import find_recordings
from .rendering import RenderedRecording, RenderJob, render_for_detection, start_render_worker
from .store import DATABASE_FILE, DetectionStore, RecordingRow, ScanIdentity


MAX_DEFAULT_WORKERS = 8
# Rendered recordings wait in memory until predicted; this bounds how many per worker.
RECORDINGS_IN_FLIGHT_PER_WORKER = 2


class DetectionError(ValueError):
    pass


@dataclass(frozen=True)
class RecordingOutcome:
    path: str
    failed: bool
    windows: int
    detections: int
    message: str | None


@dataclass(frozen=True)
class DetectionSummary:
    """found, already_done, processed, audio_seconds, and the timings describe
    this run; failed, windows, and detections describe the whole record."""

    output_dir: Path
    threshold: float
    found: int
    already_done: int
    processed: int
    failed: int
    windows: int
    detections: int
    audio_seconds: float
    wall_seconds: float
    render_seconds: float
    predict_seconds: float
    csv_problem: str | None


@dataclass
class _RunTotals:
    processed: int = 0
    audio_seconds: float = 0.0
    render_seconds: float = 0.0
    predict_seconds: float = 0.0

    def add(self, row: RecordingRow) -> None:
        self.processed += 1
        if row.status == "ok" and row.duration_s is not None:
            self.audio_seconds += row.duration_s
        self.render_seconds += row.render_seconds
        self.predict_seconds += row.predict_seconds


def default_workers() -> int:
    """Half the logical CPUs, at most eight: rendering is CPU-bound, and the rest
    stay free for prediction and for whatever else the computer is doing."""
    return max(1, min(MAX_DEFAULT_WORKERS, (os.cpu_count() or 2) // 2))


def default_output_dir(results_root: Path, input_root: Path, run_name: str) -> Path:
    return results_root / f"{input_root.resolve().name}-{run_name}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def detect_calls(
    run_dir: Path,
    input_root: Path,
    output_dir: Path,
    *,
    device: torch.device,
    workers: int,
    on_start: Callable[[int, int], None] = lambda pending, done: None,
    on_recording: Callable[[RecordingOutcome], None] = lambda outcome: None,
    clock: Callable[[], datetime] = utc_now,
) -> DetectionSummary:
    """Detect calls in every recording beneath input_root not yet stored in output_dir.

    on_start receives the number of recordings to process and the number already
    done; on_recording receives each recording once it is stored.
    """
    if workers < 1:
        raise DetectionError("workers must be at least 1")
    input_root, output_dir = input_root.resolve(), output_dir.resolve()
    if not input_root.is_dir():
        raise DetectionError(f"input folder not found: {input_root}")
    if output_dir.is_relative_to(input_root):
        raise DetectionError(
            f"the output folder must not lie inside the input folder, which holds the source recordings: {output_dir}"
        )
    saved = load_saved_run(run_dir)
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    store = DetectionStore.open(output_dir / DATABASE_FILE, _scan_identity(saved, input_root), now=_timestamp(clock()))
    with store:
        found = find_recordings(input_root)
        done = store.finished_paths()
        jobs = [RenderJob(input_root, path, saved.preprocessing) for path in found if path not in done]
        on_start(len(jobs), len(found) - len(jobs))
        use_full_precision()
        model = saved.model.to(device)
        totals = _RunTotals()
        with closing(_render_in_order(jobs, workers)) as rendered_recordings:
            for rendered in rendered_recordings:
                prediction_started = time.perf_counter()
                probabilities = _predict(model, rendered.images, device)
                row = _recording_row(rendered, time.perf_counter() - prediction_started, _timestamp(clock()))
                store.save(row, rendered.window_starts, probabilities)
                totals.add(row)
                on_recording(RecordingOutcome(
                    path=row.path,
                    failed=row.status == "failed",
                    windows=row.windows,
                    detections=int(np.count_nonzero(probabilities >= saved.threshold)),
                    message=row.message,
                ))
        csv_problem = None
        try:
            export_csv(store, output_dir)
        except ExportError as error:
            csv_problem = str(error)
        recordings = store.recording_rows()
        detections = len(store.windows(minimum_probability=saved.threshold))
    return DetectionSummary(
        output_dir=output_dir,
        threshold=saved.threshold,
        found=len(found),
        already_done=len(found) - len(jobs),
        processed=totals.processed,
        failed=sum(row.status == "failed" for row in recordings),
        windows=sum(row.windows for row in recordings),
        detections=detections,
        audio_seconds=totals.audio_seconds,
        wall_seconds=time.perf_counter() - started,
        render_seconds=totals.render_seconds,
        predict_seconds=totals.predict_seconds,
        csv_problem=csv_problem,
    )


def _scan_identity(saved: SavedRun, input_root: Path) -> ScanIdentity:
    return ScanIdentity(
        input_root=input_root.as_posix(),
        model_run=saved.name,
        model_sha256=saved.model_sha256,
        threshold=saved.threshold,
        preprocessing_sha256=saved.preprocessing.sha256,
        chunk_seconds=saved.preprocessing.audio.chunk_seconds,
    )


def _render_in_order(jobs: list[RenderJob], workers: int) -> Iterator[RenderedRecording]:
    """Rendered recordings in job order.

    With several workers, rendering runs ahead in other processes while this one
    predicts and stores.
    """
    workers = min(workers, len(jobs))
    if workers <= 1:
        yield from map(render_for_detection, jobs)
        return
    # Spawned rather than forked on every platform: forking a process that has
    # initialized CUDA is unsafe.
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=start_render_worker,
    )
    try:
        remaining = iter(jobs)
        in_flight = deque(
            executor.submit(render_for_detection, job)
            for job in islice(remaining, workers * RECORDINGS_IN_FLIGHT_PER_WORKER)
        )
        while in_flight:
            rendered = in_flight.popleft().result()
            for job in islice(remaining, 1):
                in_flight.append(executor.submit(render_for_detection, job))
            yield rendered
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _predict(model: torch.nn.Module, images: np.ndarray, device: torch.device) -> np.ndarray:
    if len(images) == 0:
        return np.empty(0)
    return predict_probabilities(model, images, device)


def _recording_row(rendered: RenderedRecording, predict_seconds: float, processed_at: str) -> RecordingRow:
    messages = ([rendered.error] if rendered.error is not None else []) + list(rendered.notes)
    return RecordingRow(
        path=rendered.path,
        recording_id=rendered.recording_id,
        started_at=None if rendered.started_at is None else rendered.started_at.isoformat(),
        utc_offset=rendered.utc_offset,
        size_bytes=rendered.size_bytes,
        audio_sha256=rendered.audio_sha256,
        duration_s=rendered.duration_s,
        windows=len(rendered.window_starts),
        status="failed" if rendered.error is not None else "ok",
        message="; ".join(messages) or None,
        render_seconds=rendered.render_seconds,
        predict_seconds=predict_seconds,
        processed_at=processed_at,
    )


def _timestamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.inference.test_detect -v`
Expected: all nine tests pass, and the run does not hang after the interruption with two workers.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/inference/detect.py tests/inference/helpers.py tests/inference/test_detect.py
git commit -m "feat: detect calls across a folder of recordings with resume"
```

---

### Task 8: The detection command, repository checks, and workflow

**Files:**
- Create: `scripts/detect_calls.py`
- Create: `results/detections/.gitkeep` (empty)
- Modify: `tests/test_repository_reproducibility.py`
- Modify: `docs/workflow.md`
- Test: `tests/inference/test_command.py`

**Interfaces:**
- Consumes: everything `frog_classifier.inference.detect` produces (Task 7), `StoreError` (Task 5), `SavedRunError` (Task 2), `select_device` from `frog_classifier.training.fit`.
- Produces: `scripts/detect_calls.py` with `main() -> int`, exit codes 0, 1, 2, and 130 as the spec states.

- [ ] **Step 1: Write the failing tests**

Create `tests/inference/test_command.py`:

```python
from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tests.inference.helpers import AUDIOMOTH_COMMENT, build_saved_run, write_recording
from tests.preprocessing.helpers import noise


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "detect_calls.py"


def _load(repo_root: Path):
    specification = importlib.util.spec_from_file_location("detect_calls_under_test", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    # The command finds the repository from its own location.
    module.__file__ = str(repo_root / "scripts" / "detect_calls.py")
    return module


def _run(module, *arguments: str) -> tuple[int, str, str]:
    output, errors = io.StringIO(), io.StringIO()
    with patch.object(sys, "argv", ["detect_calls.py", *arguments]), redirect_stdout(output), redirect_stderr(errors):
        code = module.main()
    return code, output.getvalue(), errors.getvalue()


class DetectCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo = Path(directory.name)
        self.run_dir = build_saved_run(self.repo / "models")
        site = self.repo / "raw" / "site"
        write_recording(site / "20220123_010000.WAV", noise(10, sample_rate_hz=48000), comment=AUDIOMOTH_COMMENT)
        write_recording(site / "20220123_020000.WAV", noise(5, sample_rate_hz=48000, seed=1))
        self.command = _load(self.repo)
        self.arguments = (f"models/{self.run_dir.name}", "raw/site", "--workers", "1", "--device", "cpu")

    def test_detects_into_the_default_output_folder_and_summarizes(self) -> None:
        code, output, _ = _run(self.command, *self.arguments)

        self.assertEqual(code, 0)
        self.assertIn("2 recordings to process; 0 already done.", output)
        self.assertIn("Recordings: 2 found, 0 already done, 2 processed now, 0 failed.", output)
        self.assertIn("of 3 windows at or above the threshold 0.500.", output)
        self.assertIn("Speed:", output)
        output_dir = (self.repo / "results" / "detections" / f"site-{self.run_dir.name}").resolve()
        self.assertTrue((output_dir / "detections.csv").is_file())
        self.assertIn(f"Results: {output_dir}", output)

    def test_a_second_run_processes_nothing_new_and_rewrites_the_spreadsheets(self) -> None:
        _run(self.command, *self.arguments)
        output_dir = self.repo / "results" / "detections" / f"site-{self.run_dir.name}"
        (output_dir / "recordings.csv").unlink()

        code, output, _ = _run(self.command, *self.arguments)

        self.assertEqual(code, 0)
        self.assertIn("Recordings: 2 found, 2 already done, 0 processed now, 0 failed.", output)
        self.assertNotIn("Speed:", output)
        self.assertTrue((output_dir / "recordings.csv").is_file())

    def test_a_failed_recording_exits_with_one_and_says_how_to_retry(self) -> None:
        (self.repo / "raw" / "site" / "20220123_030000.wav").write_bytes(b"not audio")

        with warnings.catch_warnings():
            # librosa warns before trying its fallback decoder; the failure is what is under test.
            warnings.simplefilter("ignore")
            code, output, errors = _run(self.command, *self.arguments)

        self.assertEqual(code, 1)
        self.assertIn("FAILED 20220123_030000.wav", output)
        self.assertIn("Run the same command again to retry them.", errors)

    def test_a_folder_that_is_not_a_run_cannot_start(self) -> None:
        code, _, errors = _run(self.command, "models", "raw/site", "--workers", "1", "--device", "cpu")

        self.assertEqual(code, 2)
        self.assertIn("Cannot start:", errors)
        self.assertIn("is not a saved run", errors)


if __name__ == "__main__":
    unittest.main()
```

In `tests/test_repository_reproducibility.py`:

- Add `"results/detections/.gitkeep",` to `PLACEHOLDERS` after `"results/data_quality/.gitkeep",`.
- Add `"results/detections/external-resnet18-20261014T000000Z/detections.sqlite",` to `GENERATED_PATHS` after `"results/data_quality/example.json",`.
- Add after `TRAINING_PACKAGE_MODULES`:

```python
INFERENCE_PACKAGE_MODULES = tuple(
    f"src/frog_classifier/inference/{name}.py"
    for name in ("__init__", "detect", "export", "recordings", "rendering", "store")
)
```

- Add `*INFERENCE_PACKAGE_MODULES,` after `*TRAINING_PACKAGE_MODULES,` in `test_project_navigation_files_are_complete_and_linked`.
- Add `"scripts/detect_calls.py",` to `OPERATIONAL_TEXT_FILES` before `"scripts/slice_audio.py",`.
- Add `"uv run python scripts/detect_calls.py",` to the command list in `test_workflow_documents_every_operational_command` after the `verify_model.py` line.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.inference.test_command tests.test_repository_reproducibility -v`
Expected: the command tests ERROR with `FileNotFoundError` for `scripts/detect_calls.py`; the repository tests FAIL for `results/detections/.gitkeep`, `scripts/detect_calls.py`, and the workflow command.

- [ ] **Step 3: Implement**

Create `scripts/detect_calls.py`:

```python
"""Detect frog calls in every recording beneath a folder, using a saved model run.

Results go to one output folder. detections.sqlite is the complete record, and
detections.csv and recordings.csv are spreadsheet copies rewritten at the end of
every run. Running the same command again continues an interrupted run and
retries recordings that failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm


INTERRUPTED = "Interrupted. Finished recordings are saved; run the same command again to continue."


def main() -> int:
    # Imported here rather than at the top: rendering workers re-run this file's
    # top-level imports when they start, and they must not load PyTorch.
    from frog_classifier.inference.detect import (
        DetectionError,
        default_output_dir,
        default_workers,
        detect_calls,
    )
    from frog_classifier.inference.store import StoreError
    from frog_classifier.training.fit import select_device
    from frog_classifier.training.saved import SavedRunError

    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("run", type=Path, help="Saved model run folder, for example models/resnet18-20261014T114432Z.")
    parser.add_argument("input", type=Path, help="Folder of recordings; .wav and .mp3 files are found in every subfolder.")
    parser.add_argument("--output", type=Path, default=None, help="Output folder. Default: results/detections/<input folder>-<run folder>.")
    parser.add_argument("--workers", type=int, default=default_workers(), help="Processes that render recordings. Default: %(default)s.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None, help="Default: the GPU when available.")
    args = parser.parse_args()

    run_dir = _resolve(repo_root, args.run)
    input_root = _resolve(repo_root, args.input)
    if args.output is not None:
        output_dir = _resolve(repo_root, args.output)
    else:
        output_dir = default_output_dir(repo_root / "results" / "detections", input_root, run_dir.name)
    try:
        device = select_device(args.device)
    except ValueError as error:
        return _cannot_start(error)

    progress = _Progress()
    try:
        summary = detect_calls(
            run_dir,
            input_root,
            output_dir,
            device=device,
            workers=args.workers,
            on_start=progress.start,
            on_recording=progress.advance,
        )
    except (DetectionError, SavedRunError, StoreError) as error:
        progress.close()
        return _cannot_start(error)
    except KeyboardInterrupt:
        progress.close()
        print(INTERRUPTED, file=sys.stderr)
        return 130
    progress.close()

    print(
        f"Recordings: {summary.found} found, {summary.already_done} already done, "
        f"{summary.processed} processed now, {summary.failed} failed."
    )
    print(
        f"Detections: {summary.detections} of {summary.windows} windows at or above "
        f"the threshold {summary.threshold:.3f}."
    )
    if summary.processed:
        print(
            f"Speed: {summary.audio_seconds / 3600:.1f} hours of audio in {summary.wall_seconds:.0f} s, "
            f"{summary.audio_seconds / summary.wall_seconds:.0f} times real time; rendering took "
            f"{summary.render_seconds:.0f} s across workers and prediction {summary.predict_seconds:.0f} s."
        )
    print(f"Results: {summary.output_dir}")
    if summary.csv_problem is not None:
        print(summary.csv_problem, file=sys.stderr)
    if summary.failed:
        print(
            f"{summary.failed} recordings failed; see recordings.csv. "
            "Run the same command again to retry them.",
            file=sys.stderr,
        )
    return 1 if summary.failed or summary.csv_problem is not None else 0


class _Progress:
    """A progress bar that appears once the number of recordings to process is known."""

    def __init__(self) -> None:
        self._bar: tqdm | None = None

    def start(self, pending: int, done: int) -> None:
        print(f"{pending} recordings to process; {done} already done.")
        self._bar = tqdm(total=pending, unit="recording")

    def advance(self, outcome) -> None:
        if outcome.failed:
            self._bar.write(f"FAILED {outcome.path}: {outcome.message}")
        self._bar.update(1)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _cannot_start(error: Exception) -> int:
    print(f"Cannot start: {error}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

Create the empty placeholder `results/detections/.gitkeep`.

In `docs/workflow.md`, add this section after the "Train and verify the detector" section and before "Verify Phase 2 acceptance":

````markdown
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
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.inference.test_command tests.test_repository_reproducibility -v`
Expected: every test passes.

- [ ] **Step 5: Commit**

```bash
git add scripts/detect_calls.py results/detections/.gitkeep tests/inference/test_command.py tests/test_repository_reproducibility.py docs/workflow.md
git commit -m "feat: add the batch detection command"
```

---

### Task 9: Real-data run and acceptance

**Files:**
- No tracked files change. Results go to `results/detections/` and the scratchpad.

**Interfaces:**
- Consumes: `scripts/detect_calls.py` (Task 8), `models/resnet18-20261014T114432Z`, `raw/external`.

- [ ] **Step 1: Full verification before touching real data**

Run:

```bash
.venv/Scripts/python.exe -m unittest discover -s tests > "$WORKSPACE/suite.log" 2>&1; tail -n 4 "$WORKSPACE/suite.log"
.venv/Scripts/python.exe -m compileall -q scripts src
/c/Users/ghimi/.local/bin/uv.exe lock --check
git diff --check
```

Expected: the suite reports `OK`, compilation prints nothing, the lock is up to date, and `git diff --check` prints nothing.

- [ ] **Step 2: Measure throughput by worker count**

Run the command over `raw/external` into scratch output folders with `--workers 1`, `4`, `8`, and `16`, one after another, recording each summary's wall time and speed line.
Expected: every run stores 618 recordings with 0 failures and the same window count; more workers are faster until rendering stops being the bottleneck.
Compare the four databases' `windows` tables: the probabilities must be identical across worker counts.
If the measurements show a clearly better default than half the logical CPUs capped at eight, record a ruling and change `MAX_DEFAULT_WORKERS` with a test-backed commit; otherwise keep it.

- [ ] **Step 3: Interrupted real run and resume**

Start the default command in the background:

```bash
.venv/Scripts/python.exe scripts/detect_calls.py models/resnet18-20261014T114432Z raw/external
```

After about a third of the recordings are stored, terminate the Python process tree abruptly.
Count the stored recordings, run the same command again, and confirm it reports that many as already done, finishes the rest, and leaves 618 recordings, no duplicate windows, and no stray Python processes.

- [ ] **Step 4: Parity with the saved run's predictions**

With a scratch script, join every row of `models/resnet18-20261014T114432Z/predictions.csv` to the detection database by recording identifier and window start, and report how many matched and the largest probability difference.
Expected: every labeled clip matches a window, and the largest difference is at most 0.0001.

- [ ] **Step 5: Inspect the output**

Report the totals from `recordings.csv` and `detections.csv` and the detections per recording hour, as a sanity check that detections concentrate in the hours the labels say frogs call.

---

### Task 10: Documentation and project status

**Files:**
- Modify: `docs/workflow.md`
- Modify: `docs/architecture.md`
- Modify: `docs/outline.md`
- Modify: `README.md`
- Modify: `roadmap.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update the documents**

- `docs/workflow.md`: add the measured throughput, worker comparison, and parity result to "Detect calls in recordings".
- `docs/architecture.md`: add a batch-detection section describing the `frog_classifier.inference` modules and their boundaries (no PyTorch in workers, one transaction per recording, the scan identity), the shared `render_waveform_images` and `training/saved.py`, the `results/detections/` storage contract with the three files, and the detection command in the component diagram.
- `docs/outline.md` and `README.md`: describe batch detection as working, with the command.
- `roadmap.md`: mark Phase 5 complete with its verified acceptance (repeatable detections, resume after interruption, provenance in every row, measured throughput, parity with the saved run), record the SQLite decision, and mark Phase 6 `Status: Next`.
- `AGENTS.md`: Phases 1, 2, 4, and 5 complete; Phase 6 next.

- [ ] **Step 2: Verify and commit**

Run the full suite, `git diff --check`, and the em dash scan over every changed file, then:

```bash
git add docs/workflow.md docs/architecture.md docs/outline.md README.md roadmap.md AGENTS.md
git commit -m "docs: document batch detection and its measured results"
```
