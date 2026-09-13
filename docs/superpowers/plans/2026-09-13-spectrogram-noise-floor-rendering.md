# Spectrogram Noise-Floor Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render every five-second Mel spectrogram as decibels above its own recording's noise floor, through one tested package used by slicing and verification, and regenerate all images without losing a single human label.

**Architecture:** A new `frog_classifier.preprocessing` package holds pure array functions (Mel decibels, noise floor, chunk image, PNG encoding), a recording renderer that yields chunk PNG bytes, and a display colour map. The configuration loader moves to schema version 2 with a `normalization` section. Three thin scripts slice recordings, move freshly rendered images over their labeled counterparts by name, and re-render a sample from raw audio to prove stored bytes match the contract.

**Tech Stack:** Python 3.13, NumPy, librosa, Pillow, soundfile (already locked as a librosa dependency, used only in tests), Matplotlib (labeler display only), unittest.

**Spec:** `docs/superpowers/specs/2026-09-13-spectrogram-noise-floor-rendering-design.md`

## Global Constraints

- Python 3.13 only; `pyproject.toml` and `uv.lock` do not change. No new dependencies.
- Run tests with `uv run python -m unittest ...`. If `uv` is not on the PATH in your shell, use `.venv/Scripts/python.exe -m unittest ...` from the repository root; the two are equivalent here.
- Every test uses synthetic audio or images in a temporary directory. Never read `raw/`, `processed/`, or `labeled/` from a test.
- Never delete, rename, or rewrite anything under `raw/` or `labeled/` except through `scripts/sync_labeled_images.py` in Task 9.
- Files must use LF line endings. When writing files from Python use `newline="\n"` or `write_bytes`. Check with `git diff --check` before every commit.
- Markdown: one full sentence per physical line. Never use the em dash character; use `-`.
- Do not commit generated data: check `git status --short` before every commit.
- Do not add a co-author line to commit messages.
- Work on a branch named `feature/spectrogram-noise-floor-rendering` created from `main`. Tasks 1 to 8 may run in a worktree. Task 9 must run in the main checkout `D:\projects\frog-classifier`, because the recordings and labels are ignored by Git and exist only there.
- The band is 400 Hz to 4,000 Hz, the chunk is 5 seconds at 22,050 Hz, hop 512 and 2048-point FFT, 128 Mel bands. A chunk renders to an image 216 pixels wide and 128 pixels tall.
- Normalization defaults: `reference = "recording"`, `noise_floor_percentile = 50`, `db_floor = 0`, `db_ceiling = 30`.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/frog_classifier/preprocessing/__init__.py` | Public exports of the preprocessing package. |
| `src/frog_classifier/preprocessing/spectrogram.py` | Pure array functions: Mel decibels, noise floor, chunk image, PNG encode and decode. |
| `src/frog_classifier/preprocessing/recording.py` | Load one recording, compute its floor, yield rendered chunks. |
| `src/frog_classifier/preprocessing/display.py` | Colour-map a grayscale PNG for the labelers. |
| `src/frog_classifier/data/config.py` | Schema 2 loader with the `normalization` section. |
| `src/frog_classifier/data/__init__.py` | Export `NormalizationConfig`. |
| `src/frog_classifier/data/reporting.py` | Provenance note points at the verification command. |
| `config/preprocessing.toml` | Schema 2 contract. |
| `scripts/slice_audio.py` | Thin command over `render_recording`; no preprocessing overrides. |
| `scripts/sync_labeled_images.py` | Move fresh images over labeled images by name. |
| `scripts/verify_spectrograms.py` | Re-render a sample from raw audio and compare bytes. |
| `scripts/label_frontend.py`, `scripts/label_spectrograms.py` | Display grayscale images through viridis. |
| `tests/preprocessing/helpers.py` | Synthetic tones, WAV writer, Mel band lookup. |
| `tests/preprocessing/test_spectrogram.py`, `test_recording.py`, `test_display.py` | Package tests. |
| `tests/test_slice_audio.py`, `test_sync_labeled_images.py`, `test_verify_spectrograms.py` | Script tests. |
| `tests/data/test_config.py`, `tests/data/helpers.py` | Schema 2 configuration tests. |
| `tests/test_repository_reproducibility.py` | Knows the new modules, scripts, and decision record. |
| `README.md`, `docs/outline.md`, `docs/architecture.md`, `docs/workflow.md`, `roadmap.md` | Documentation. |

---

### Task 1: Pure spectrogram functions

**Files:**
- Create: `src/frog_classifier/preprocessing/__init__.py`
- Create: `src/frog_classifier/preprocessing/spectrogram.py`
- Create: `tests/preprocessing/__init__.py`
- Create: `tests/preprocessing/helpers.py`
- Create: `tests/preprocessing/test_spectrogram.py`

**Interfaces:**
- Produces: `mel_decibels(waveform, *, sample_rate_hz, n_mels, fmin_hz, fmax_hz, power, n_fft, hop_length) -> np.ndarray` of shape `(n_mels, frames)` in absolute decibels.
- Produces: `noise_floor(mel_db, percentile) -> np.ndarray` of shape `(n_mels,)`.
- Produces: `chunk_image(mel_db, floor, *, db_floor, db_ceiling) -> np.ndarray` of dtype `uint8`, lowest band as the bottom row.
- Produces: `encode_png(image) -> bytes` and `decode_png(png_bytes) -> np.ndarray`.
- Produces test helpers: `tone(frequency_hz, seconds, *, amplitude=0.5, sample_rate_hz=22050)`, `silence(seconds, *, sample_rate_hz=22050)`, `write_wav(path, waveform, *, sample_rate_hz=22050) -> Path`, `mel_band_for(frequency_hz) -> int`.

- [ ] **Step 1: Create the test package and helpers**

Create `tests/preprocessing/__init__.py` as an empty file.

Create `tests/preprocessing/helpers.py`:

```python
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile


SAMPLE_RATE_HZ = 22050
N_MELS = 128
FMIN_HZ = 400
FMAX_HZ = 4000


def tone(
    frequency_hz: float,
    seconds: float,
    *,
    amplitude: float = 0.5,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> np.ndarray:
    samples = int(round(seconds * sample_rate_hz))
    times = np.arange(samples, dtype=np.float64) / sample_rate_hz
    return (amplitude * np.sin(2.0 * np.pi * frequency_hz * times)).astype(np.float32)


def silence(seconds: float, *, sample_rate_hz: int = SAMPLE_RATE_HZ) -> np.ndarray:
    return np.zeros(int(round(seconds * sample_rate_hz)), dtype=np.float32)


def write_wav(
    path: Path,
    waveform: np.ndarray,
    *,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, waveform, sample_rate_hz, subtype="PCM_16")
    return path


def mel_band_for(frequency_hz: float) -> int:
    """Index of the Mel band whose centre is closest to the frequency."""
    centres = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN_HZ, fmax=FMAX_HZ)
    return int(np.argmin(np.abs(centres - frequency_hz)))
```

- [ ] **Step 2: Write the failing tests**

Create `tests/preprocessing/test_spectrogram.py`:

```python
from __future__ import annotations

import unittest

import numpy as np

from frog_classifier.preprocessing.spectrogram import (
    chunk_image,
    decode_png,
    encode_png,
    mel_decibels,
    noise_floor,
)
from tests.preprocessing.helpers import (
    SAMPLE_RATE_HZ,
    mel_band_for,
    silence,
    tone,
)


MEL_SETTINGS = dict(
    sample_rate_hz=SAMPLE_RATE_HZ,
    n_mels=128,
    fmin_hz=400,
    fmax_hz=4000,
    power=2.0,
    n_fft=2048,
    hop_length=512,
)


class MelDecibelsTests(unittest.TestCase):
    def test_five_second_chunk_has_128_bands_and_216_frames(self) -> None:
        mel_db = mel_decibels(silence(5), **MEL_SETTINGS)

        self.assertEqual(mel_db.shape, (128, 216))

    def test_silence_is_the_decibel_floor_everywhere(self) -> None:
        mel_db = mel_decibels(silence(5), **MEL_SETTINGS)

        self.assertTrue(np.allclose(mel_db, -100.0))


class NoiseFloorTests(unittest.TestCase):
    def test_returns_requested_percentile_per_band(self) -> None:
        mel_db = np.array([[0.0, 10.0, 20.0, 30.0], [5.0, 5.0, 5.0, 100.0]])

        self.assertEqual(noise_floor(mel_db, 50).tolist(), [15.0, 5.0])
        self.assertEqual(noise_floor(mel_db, 0).tolist(), [0.0, 5.0])

    def test_rejects_percentile_outside_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "percentile"):
            noise_floor(np.zeros((2, 3)), 101)

    def test_rejects_empty_frames(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame"):
            noise_floor(np.zeros((2, 0)), 50)


class ChunkImageTests(unittest.TestCase):
    def test_maps_floor_to_black_ceiling_to_white_and_clips_outside(self) -> None:
        mel_db = np.array([[-5.0], [0.0], [12.0], [30.0], [40.0]])
        floor = np.zeros(5)

        image = chunk_image(mel_db, floor, db_floor=0.0, db_ceiling=30.0)

        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image[:, 0].tolist(), [255, 255, 102, 0, 0])

    def test_lowest_band_is_the_bottom_row(self) -> None:
        mel_db = np.zeros((4, 3))
        mel_db[0, :] = 30.0
        floor = np.zeros(4)

        image = chunk_image(mel_db, floor, db_floor=0.0, db_ceiling=30.0)

        self.assertEqual(image[-1, :].tolist(), [255, 255, 255])
        self.assertEqual(image[0, :].tolist(), [0, 0, 0])

    def test_subtracts_the_floor_per_band(self) -> None:
        mel_db = np.array([[40.0], [40.0]])
        floor = np.array([40.0, 10.0])

        image = chunk_image(mel_db, floor, db_floor=0.0, db_ceiling=30.0)

        self.assertEqual(image[:, 0].tolist(), [255, 0])

    def test_rejects_ceiling_not_above_floor(self) -> None:
        with self.assertRaisesRegex(ValueError, "ceiling"):
            chunk_image(np.zeros((2, 2)), np.zeros(2), db_floor=10.0, db_ceiling=10.0)

    def test_rejects_floor_with_wrong_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "band"):
            chunk_image(np.zeros((2, 2)), np.zeros(3), db_floor=0.0, db_ceiling=30.0)

    def test_stationary_tone_renders_near_black(self) -> None:
        recording = tone(1000.0, 30)
        floor = noise_floor(mel_decibels(recording, **MEL_SETTINGS), 50)
        chunk = mel_decibels(recording[: 5 * SAMPLE_RATE_HZ], **MEL_SETTINGS)

        image = chunk_image(chunk, floor, db_floor=0.0, db_ceiling=30.0)

        # One decibel above the floor is about 8 grey levels.
        self.assertLessEqual(int(image.max()), 8)

    def test_transient_tone_renders_bright_in_its_band_only(self) -> None:
        recording = np.concatenate([silence(5), tone(1000.0, 5), silence(5)])
        floor = noise_floor(mel_decibels(recording, **MEL_SETTINGS), 50)
        chunk_size = 5 * SAMPLE_RATE_HZ
        quiet = chunk_image(
            mel_decibels(recording[:chunk_size], **MEL_SETTINGS),
            floor,
            db_floor=0.0,
            db_ceiling=30.0,
        )
        loud = chunk_image(
            mel_decibels(recording[chunk_size : 2 * chunk_size], **MEL_SETTINGS),
            floor,
            db_floor=0.0,
            db_ceiling=30.0,
        )

        self.assertEqual(int(quiet.max()), 0)
        self.assertEqual(int(loud.max()), 255)
        brightest_row = int(np.argmax(loud.max(axis=1)))
        expected_row = 127 - mel_band_for(1000.0)
        self.assertLessEqual(abs(brightest_row - expected_row), 2)


class PngTests(unittest.TestCase):
    def test_round_trips_and_is_deterministic(self) -> None:
        image = np.arange(128 * 216, dtype=np.uint32).reshape(128, 216) % 256
        image = image.astype(np.uint8)

        first = encode_png(image)
        second = encode_png(image)

        self.assertEqual(first, second)
        self.assertTrue(np.array_equal(decode_png(first), image))

    def test_rejects_non_grayscale_arrays(self) -> None:
        with self.assertRaisesRegex(ValueError, "uint8"):
            encode_png(np.zeros((2, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run python -m unittest tests.preprocessing.test_spectrogram -v`
Expected: FAIL at import with `ModuleNotFoundError: No module named 'frog_classifier.preprocessing'`.

- [ ] **Step 4: Write the implementation**

Create `src/frog_classifier/preprocessing/spectrogram.py`:

```python
from __future__ import annotations

import io

import librosa
import numpy as np
from PIL import Image


def mel_decibels(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int,
    n_mels: int,
    fmin_hz: int,
    fmax_hz: int,
    power: float,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """Mel power spectrogram in absolute decibels with shape (n_mels, frames).

    The reference is 1.0 and no top_db clipping is applied, so values are
    comparable between a whole recording and any chunk of it.
    """
    mel_power = librosa.feature.melspectrogram(
        y=np.asarray(waveform, dtype=np.float32),
        sr=sample_rate_hz,
        n_mels=n_mels,
        fmin=fmin_hz,
        fmax=fmax_hz,
        power=power,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return librosa.power_to_db(mel_power, ref=1.0, top_db=None)


def noise_floor(mel_db: np.ndarray, percentile: int) -> np.ndarray:
    """One floor value per band: the requested percentile across all frames."""
    if mel_db.ndim != 2 or mel_db.shape[1] == 0:
        raise ValueError("mel_db must have shape (bands, frames) with at least one frame")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    return np.percentile(mel_db, percentile, axis=1)


def chunk_image(
    mel_db: np.ndarray,
    floor: np.ndarray,
    *,
    db_floor: float,
    db_ceiling: float,
) -> np.ndarray:
    """8-bit image of decibels above the floor with the lowest band at the bottom."""
    if db_ceiling <= db_floor:
        raise ValueError("db_ceiling must exceed db_floor")
    if floor.shape != (mel_db.shape[0],):
        raise ValueError("floor must hold exactly one value per band")
    relative = mel_db - floor[:, np.newaxis]
    unit = np.clip((relative - db_floor) / (db_ceiling - db_floor), 0.0, 1.0)
    pixels = np.rint(unit * 255.0).astype(np.uint8)
    return pixels[::-1, :]


def encode_png(image: np.ndarray) -> bytes:
    """Lossless 8-bit grayscale PNG bytes with no metadata chunks."""
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("image must be a two-dimensional uint8 array")
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(buffer, format="PNG")
    return buffer.getvalue()


def decode_png(png_bytes: bytes) -> np.ndarray:
    """Grayscale uint8 array from PNG bytes."""
    with Image.open(io.BytesIO(png_bytes)) as image:
        return np.array(image.convert("L"), dtype=np.uint8)
```

Create `src/frog_classifier/preprocessing/__init__.py`:

```python
from .spectrogram import (
    chunk_image,
    decode_png,
    encode_png,
    mel_decibels,
    noise_floor,
)

__all__ = [
    "chunk_image",
    "decode_png",
    "encode_png",
    "mel_decibels",
    "noise_floor",
]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.preprocessing.test_spectrogram -v`
Expected: `Ran 14 tests` and `OK`.

If `test_stationary_tone_renders_near_black` fails with a small maximum such as 9 or 10, the chunk edge frames or floating point rounding are slightly off; loosen the bound to 12 and add a comment giving the measured value. Do not loosen beyond 12.

- [ ] **Step 6: Run the whole suite and commit**

Run: `uv run python -m unittest discover -s tests`
Expected: all tests pass (86 existing plus 14 new).

```bash
git add src/frog_classifier/preprocessing tests/preprocessing
git diff --cached --check
git commit -m "feat: add pure spectrogram rendering functions"
```

---

### Task 2: Schema 2 configuration

**Files:**
- Modify: `src/frog_classifier/data/config.py`
- Modify: `src/frog_classifier/data/__init__.py`
- Modify: `config/preprocessing.toml`
- Modify: `tests/data/helpers.py`
- Modify: `tests/data/test_config.py`
- Modify: `tests/test_slice_audio.py`

**Interfaces:**
- Produces: `NormalizationConfig(reference: str, noise_floor_percentile: int, db_floor: float, db_ceiling: float)`.
- Produces: `RenderingConfig(format: str, bit_depth: int, low_frequency_at_bottom: bool)`.
- Produces: `PreprocessingConfig` gains the field `normalization: NormalizationConfig` between `spectrogram` and `rendering`; `schema_version` is 2.
- Consumers in later tasks read `config.audio.sample_rate_hz`, `config.audio.chunk_seconds`, `config.spectrogram.*`, `config.normalization.noise_floor_percentile`, `config.normalization.db_floor`, `config.normalization.db_ceiling`.

Note: after this task and until Task 4 lands, `scripts/slice_audio.py` cannot run because it still reads the removed figure fields. Task 4 rewrites it. Do not run the slicer between the two commits.

- [ ] **Step 1: Update the test configuration in both helpers**

In `tests/data/helpers.py` and in `tests/data/test_config.py`, replace the whole `VALID_CONFIG = b'''...'''` literal with:

```python
VALID_CONFIG = b'''schema_version = 2

[audio]
sample_rate_hz = 22050
mono = true
chunk_seconds = 5
overlap_seconds = 0
drop_incomplete_final_chunk = true

[spectrogram]
kind = "mel"
n_mels = 128
fmin_hz = 400
fmax_hz = 4000
power = 2.0
n_fft = 2048
hop_length = 512

[normalization]
reference = "recording"
noise_floor_percentile = 50
db_floor = 0
db_ceiling = 30

[rendering]
format = "png"
bit_depth = 8
low_frequency_at_bottom = true

[classes]
litoria_aurea = 1
non_target = 0
'''
```

- [ ] **Step 2: Rewrite the configuration tests**

Replace the class body of `LoadPreprocessingConfigTests` in `tests/data/test_config.py` with:

```python
class LoadPreprocessingConfigTests(unittest.TestCase):
    def load(self, content: bytes):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "preprocessing.toml"
            path.write_bytes(content)
            return load_preprocessing_config(path), path.resolve()

    def test_loads_every_tracked_preprocessing_value(self) -> None:
        config, source_path = self.load(VALID_CONFIG)

        self.assertEqual(config.schema_version, 2)
        self.assertEqual(config.audio.sample_rate_hz, 22050)
        self.assertTrue(config.audio.mono)
        self.assertEqual(config.audio.chunk_seconds, 5)
        self.assertEqual(config.audio.overlap_seconds, 0)
        self.assertTrue(config.audio.drop_incomplete_final_chunk)
        self.assertEqual(config.spectrogram.kind, "mel")
        self.assertEqual(config.spectrogram.n_mels, 128)
        self.assertEqual(config.spectrogram.fmin_hz, 400)
        self.assertEqual(config.spectrogram.fmax_hz, 4000)
        self.assertEqual(config.spectrogram.power, 2.0)
        self.assertEqual(config.spectrogram.n_fft, 2048)
        self.assertEqual(config.spectrogram.hop_length, 512)
        self.assertEqual(config.normalization.reference, "recording")
        self.assertEqual(config.normalization.noise_floor_percentile, 50)
        self.assertEqual(config.normalization.db_floor, 0.0)
        self.assertEqual(config.normalization.db_ceiling, 30.0)
        self.assertEqual(config.rendering.format, "png")
        self.assertEqual(config.rendering.bit_depth, 8)
        self.assertTrue(config.rendering.low_frequency_at_bottom)
        self.assertEqual(dict(config.classes), {"litoria_aurea": 1, "non_target": 0})
        self.assertEqual(config.source_path, source_path)
        self.assertEqual(config.sha256, hashlib.sha256(VALID_CONFIG).hexdigest())

    def test_rejects_schema_version_one_and_asks_for_regeneration(self) -> None:
        with self.assertRaisesRegex(ConfigError, "regenerated"):
            self.load(VALID_CONFIG.replace(b"schema_version = 2", b"schema_version = 1"))

    def test_rejects_missing_normalization_section(self) -> None:
        invalid_config = VALID_CONFIG.replace(
            b'''[normalization]
reference = "recording"
noise_floor_percentile = 50
db_floor = 0
db_ceiling = 30

''',
            b"",
        )

        with self.assertRaisesRegex(ConfigError, "normalization"):
            self.load(invalid_config)

    def test_rejects_missing_rendering_section(self) -> None:
        invalid_config = VALID_CONFIG.replace(
            b'''[rendering]
format = "png"
bit_depth = 8
low_frequency_at_bottom = true

''',
            b"",
        )

        with self.assertRaisesRegex(ConfigError, "rendering"):
            self.load(invalid_config)

    def test_rejects_unknown_normalization_reference(self) -> None:
        with self.assertRaisesRegex(ConfigError, "reference"):
            self.load(VALID_CONFIG.replace(b'reference = "recording"', b'reference = "chunk"'))

    def test_rejects_percentile_outside_range(self) -> None:
        for value in (b"-1", b"101"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ConfigError, "noise_floor_percentile"):
                    self.load(VALID_CONFIG.replace(
                        b"noise_floor_percentile = 50",
                        b"noise_floor_percentile = " + value,
                    ))

    def test_rejects_ceiling_not_above_floor(self) -> None:
        with self.assertRaisesRegex(ConfigError, "db_ceiling"):
            self.load(VALID_CONFIG.replace(b"db_ceiling = 30", b"db_ceiling = 0"))

    def test_rejects_nonfinite_normalization_values(self) -> None:
        fields = (("db_floor", b"db_floor = 0"), ("db_ceiling", b"db_ceiling = 30"))
        for field, original in fields:
            for value in ("nan", "inf", "-inf"):
                with self.subTest(field=field, value=value):
                    invalid = VALID_CONFIG.replace(
                        original,
                        f"{field} = {value}".encode("ascii"),
                    )

                    with self.assertRaisesRegex(ConfigError, field):
                        self.load(invalid)

    def test_rejects_unsupported_bit_depth(self) -> None:
        with self.assertRaisesRegex(ConfigError, "bit_depth"):
            self.load(VALID_CONFIG.replace(b"bit_depth = 8", b"bit_depth = 16"))

    def test_rejects_high_frequency_at_bottom(self) -> None:
        with self.assertRaisesRegex(ConfigError, "low_frequency_at_bottom"):
            self.load(VALID_CONFIG.replace(
                b"low_frequency_at_bottom = true",
                b"low_frequency_at_bottom = false",
            ))

    def test_rejects_overlap_equal_to_chunk_duration(self) -> None:
        with self.assertRaisesRegex(ConfigError, "overlap"):
            self.load(VALID_CONFIG.replace(b"overlap_seconds = 0", b"overlap_seconds = 5"))

    def test_rejects_frequency_above_nyquist(self) -> None:
        with self.assertRaisesRegex(ConfigError, "frequency"):
            self.load(VALID_CONFIG.replace(b"fmax_hz = 4000", b"fmax_hz = 12000"))

    def test_rejects_noncanonical_class_mapping(self) -> None:
        with self.assertRaisesRegex(ConfigError, "classes"):
            self.load(VALID_CONFIG.replace(b"litoria_aurea = 1", b"litoria_aurea = 0"))

    def test_rejects_nonfinite_spectrogram_float_values(self) -> None:
        for value in ("nan", "inf", "-inf"):
            with self.subTest(field="power", value=value):
                invalid = VALID_CONFIG.replace(
                    b"power = 2.0",
                    f"power = {value}".encode("ascii"),
                )

                with self.assertRaisesRegex(ConfigError, "power"):
                    self.load(invalid)
```

- [ ] **Step 3: Remove the obsolete slicer integration test**

Replace the entire content of `tests/test_slice_audio.py` with the following, which keeps only the guard that the slicer carries no preprocessing defaults. Task 4 replaces this file again with end-to-end tests.

```python
from __future__ import annotations

import dataclasses
import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "slice_audio.py"


class SliceAudioPreprocessingIntegrationTests(unittest.TestCase):
    def test_slicer_config_has_no_preprocessing_defaults(self) -> None:
        """The tracked TOML is the only source of preprocessing values.

        Stale dataclass defaults once diverged from the command-line defaults
        and hid which frequency band actually produced the spectrograms.
        """
        module = self._load_module()
        fields = dataclasses.fields(module.Config)
        defaulted = sorted(
            field.name
            for field in fields
            if field.default is not dataclasses.MISSING
            or field.default_factory is not dataclasses.MISSING
        )
        self.assertEqual(defaulted, [])

    def _load_module(self):
        module_name = f"slice_audio_test_{id(self)}"
        specification = importlib.util.spec_from_file_location(
            module_name,
            SCRIPT_PATH,
        )
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)
        specification.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run the configuration tests to verify they fail**

Run: `uv run python -m unittest tests.data.test_config -v`
Expected: FAIL. `test_loads_every_tracked_preprocessing_value` raises `ConfigError` about top-level keys because `normalization` is unexpected, and the new rejection tests fail for the same reason rather than the expected message.

- [ ] **Step 5: Rewrite the loader**

Replace the entire content of `src/frog_classifier/data/config.py` with:

```python
from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


SUPPORTED_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class AudioConfig:
    sample_rate_hz: int
    mono: bool
    chunk_seconds: int
    overlap_seconds: int
    drop_incomplete_final_chunk: bool


@dataclass(frozen=True)
class SpectrogramConfig:
    kind: str
    n_mels: int
    fmin_hz: int
    fmax_hz: int
    power: float
    n_fft: int
    hop_length: int


@dataclass(frozen=True)
class NormalizationConfig:
    reference: str
    noise_floor_percentile: int
    db_floor: float
    db_ceiling: float


@dataclass(frozen=True)
class RenderingConfig:
    format: str
    bit_depth: int
    low_frequency_at_bottom: bool


@dataclass(frozen=True)
class PreprocessingConfig:
    schema_version: int
    audio: AudioConfig
    spectrogram: SpectrogramConfig
    normalization: NormalizationConfig
    rendering: RenderingConfig
    classes: Mapping[str, int]
    source_path: Path
    sha256: str


class ConfigError(ValueError):
    pass


def _section(data: dict[str, object], name: str, keys: set[str]) -> dict[str, object]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing or invalid {name} section")
    if set(value) != keys:
        raise ConfigError(f"invalid {name} keys: {sorted(value)}")
    return value


def _value(data: dict[str, object], key: str, expected: type) -> object:
    value = data[key]
    if expected is int and (not isinstance(value, int) or isinstance(value, bool)):
        raise ConfigError(f"{key} must be an integer")
    if expected is float and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise ConfigError(f"{key} must be numeric")
    if expected is float and not math.isfinite(float(value)):
        raise ConfigError(f"{key} must be finite")
    if expected is bool and not isinstance(value, bool):
        raise ConfigError(f"{key} must be a Boolean")
    if expected is str and not isinstance(value, str):
        raise ConfigError(f"{key} must be text")
    return value


def load_preprocessing_config(path: Path) -> PreprocessingConfig:
    source = path.read_bytes()
    try:
        data = tomllib.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"invalid preprocessing TOML: {error}") from error

    expected_top_level = {
        "schema_version", "audio", "spectrogram", "normalization", "rendering", "classes",
    }
    if set(data) != expected_top_level:
        missing = sorted(expected_top_level - set(data))
        unexpected = sorted(set(data) - expected_top_level)
        raise ConfigError(
            f"invalid top-level keys: {sorted(data)}; "
            f"missing: {missing}; unexpected: {unexpected}"
        )
    if _value(data, "schema_version", int) != SUPPORTED_SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version must be {SUPPORTED_SCHEMA_VERSION}; spectrograms rendered "
            "under an earlier schema must be regenerated"
        )

    audio_data = _section(data, "audio", {
        "sample_rate_hz", "mono", "chunk_seconds", "overlap_seconds",
        "drop_incomplete_final_chunk",
    })
    spectrogram_data = _section(data, "spectrogram", {
        "kind", "n_mels", "fmin_hz", "fmax_hz", "power", "n_fft", "hop_length",
    })
    normalization_data = _section(data, "normalization", {
        "reference", "noise_floor_percentile", "db_floor", "db_ceiling",
    })
    rendering_data = _section(data, "rendering", {
        "format", "bit_depth", "low_frequency_at_bottom",
    })
    classes_data = _section(data, "classes", {"litoria_aurea", "non_target"})

    audio = AudioConfig(
        sample_rate_hz=int(_value(audio_data, "sample_rate_hz", int)),
        mono=bool(_value(audio_data, "mono", bool)),
        chunk_seconds=int(_value(audio_data, "chunk_seconds", int)),
        overlap_seconds=int(_value(audio_data, "overlap_seconds", int)),
        drop_incomplete_final_chunk=bool(_value(audio_data, "drop_incomplete_final_chunk", bool)),
    )
    spectrogram = SpectrogramConfig(
        kind=str(_value(spectrogram_data, "kind", str)),
        n_mels=int(_value(spectrogram_data, "n_mels", int)),
        fmin_hz=int(_value(spectrogram_data, "fmin_hz", int)),
        fmax_hz=int(_value(spectrogram_data, "fmax_hz", int)),
        power=float(_value(spectrogram_data, "power", float)),
        n_fft=int(_value(spectrogram_data, "n_fft", int)),
        hop_length=int(_value(spectrogram_data, "hop_length", int)),
    )
    normalization = NormalizationConfig(
        reference=str(_value(normalization_data, "reference", str)),
        noise_floor_percentile=int(_value(normalization_data, "noise_floor_percentile", int)),
        db_floor=float(_value(normalization_data, "db_floor", float)),
        db_ceiling=float(_value(normalization_data, "db_ceiling", float)),
    )
    rendering = RenderingConfig(
        format=str(_value(rendering_data, "format", str)),
        bit_depth=int(_value(rendering_data, "bit_depth", int)),
        low_frequency_at_bottom=bool(_value(rendering_data, "low_frequency_at_bottom", bool)),
    )
    classes = {
        "litoria_aurea": int(_value(classes_data, "litoria_aurea", int)),
        "non_target": int(_value(classes_data, "non_target", int)),
    }

    if audio.sample_rate_hz <= 0 or audio.chunk_seconds <= 0:
        raise ConfigError("audio dimensions must be positive")
    if not 0 <= audio.overlap_seconds < audio.chunk_seconds:
        raise ConfigError("overlap must be nonnegative and shorter than the chunk")
    if not audio.mono or audio.overlap_seconds != 0 or not audio.drop_incomplete_final_chunk:
        raise ConfigError("audio settings must match the supported fixed-window contract")
    if not 0 <= spectrogram.fmin_hz < spectrogram.fmax_hz <= audio.sample_rate_hz / 2:
        raise ConfigError("frequency range must not exceed Nyquist")
    if min(
        spectrogram.n_mels,
        spectrogram.n_fft,
        spectrogram.hop_length,
        spectrogram.power,
    ) <= 0:
        raise ConfigError("spectrogram dimensions and power must be positive")
    if spectrogram.kind != "mel" or rendering.format != "png":
        raise ConfigError("only mel spectrograms rendered as png are supported")
    if normalization.reference != "recording":
        raise ConfigError("normalization reference must be recording")
    if not 0 <= normalization.noise_floor_percentile <= 100:
        raise ConfigError("noise_floor_percentile must be between 0 and 100")
    if normalization.db_ceiling <= normalization.db_floor:
        raise ConfigError("db_ceiling must exceed db_floor")
    if rendering.bit_depth != 8:
        raise ConfigError("bit_depth must be 8")
    if not rendering.low_frequency_at_bottom:
        raise ConfigError("low_frequency_at_bottom must be true")
    if classes != {"litoria_aurea": 1, "non_target": 0}:
        raise ConfigError("classes must match the canonical mapping")

    return PreprocessingConfig(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        audio=audio,
        spectrogram=spectrogram,
        normalization=normalization,
        rendering=rendering,
        classes=MappingProxyType(classes),
        source_path=path.resolve(),
        sha256=hashlib.sha256(source).hexdigest(),
    )
```

In `src/frog_classifier/data/__init__.py`, add `NormalizationConfig,` to the `from .config import (...)` list after `ConfigError,` and add `"NormalizationConfig",` to `__all__` after `"ManifestValidationError",`.

- [ ] **Step 6: Update the tracked configuration**

Replace the entire content of `config/preprocessing.toml` with:

```toml
schema_version = 2

[audio]
sample_rate_hz = 22050
mono = true
chunk_seconds = 5
overlap_seconds = 0
drop_incomplete_final_chunk = true

[spectrogram]
kind = "mel"
n_mels = 128
fmin_hz = 400
fmax_hz = 4000
power = 2.0
n_fft = 2048
hop_length = 512

[normalization]
reference = "recording"
noise_floor_percentile = 50
db_floor = 0
db_ceiling = 30

[rendering]
format = "png"
bit_depth = 8
low_frequency_at_bottom = true

[classes]
litoria_aurea = 1
non_target = 0
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.data.test_config -v`
Expected: `Ran 14 tests` and `OK`.

Run: `uv run python -m unittest discover -s tests`
Expected: all pass. The manifest, reporting, baseline, and CLI tests read `VALID_CONFIG` from `tests/data/helpers.py`, which now carries schema 2.

- [ ] **Step 8: Commit**

```bash
git add src/frog_classifier/data/config.py src/frog_classifier/data/__init__.py config/preprocessing.toml tests/data/helpers.py tests/data/test_config.py tests/test_slice_audio.py
git diff --cached --check
git commit -m "feat: move the preprocessing contract to schema 2 with normalization"
```

---

### Task 3: Recording renderer

**Files:**
- Create: `src/frog_classifier/preprocessing/recording.py`
- Modify: `src/frog_classifier/preprocessing/__init__.py`
- Create: `tests/preprocessing/test_recording.py`

**Interfaces:**
- Consumes: `PreprocessingConfig` from Task 2; `mel_decibels`, `noise_floor`, `chunk_image`, `encode_png` from Task 1.
- Produces: `RenderedChunk(start_s: int, png_bytes: bytes)` frozen dataclass.
- Produces: `load_waveform(path: Path, config: PreprocessingConfig) -> np.ndarray`.
- Produces: `render_waveform(waveform: np.ndarray, config: PreprocessingConfig) -> Iterator[RenderedChunk]`.
- Produces: `render_recording(path: Path, config: PreprocessingConfig) -> Iterator[RenderedChunk]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/preprocessing/test_recording.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile

from frog_classifier.preprocessing.recording import (
    RenderedChunk,
    render_recording,
    render_waveform,
)
from frog_classifier.preprocessing.spectrogram import decode_png
from tests.data.helpers import load_test_config
from tests.preprocessing.helpers import silence, tone, write_wav


class RenderRecordingTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = load_test_config(self.root)

    def test_yields_one_chunk_per_complete_window_in_order(self) -> None:
        waveform = np.concatenate([silence(5), tone(1000.0, 5), silence(7)])
        path = write_wav(self.root / "raw" / "site" / "recording.wav", waveform)

        chunks = list(render_recording(path, self.config))

        self.assertEqual([chunk.start_s for chunk in chunks], [0, 5, 10])
        for chunk in chunks:
            self.assertIsInstance(chunk, RenderedChunk)
            image = decode_png(chunk.png_bytes)
            self.assertEqual(image.shape, (128, 216))
            self.assertEqual(image.dtype, np.uint8)

    def test_uses_the_whole_recording_as_the_noise_floor(self) -> None:
        waveform = np.concatenate([silence(5), tone(1000.0, 5), silence(7)])
        path = write_wav(self.root / "raw" / "recording.wav", waveform)

        chunks = list(render_recording(path, self.config))

        self.assertEqual(int(decode_png(chunks[0].png_bytes).max()), 0)
        self.assertEqual(int(decode_png(chunks[1].png_bytes).max()), 255)
        self.assertEqual(int(decode_png(chunks[2].png_bytes).max()), 0)

    def test_file_shorter_than_one_chunk_yields_nothing(self) -> None:
        path = write_wav(self.root / "raw" / "short.wav", tone(1000.0, 3))

        self.assertEqual(list(render_recording(path, self.config)), [])

    def test_render_waveform_matches_render_recording(self) -> None:
        waveform = np.concatenate([silence(2), tone(800.0, 3), silence(5)])
        path = write_wav(self.root / "raw" / "recording.wav", waveform)

        from_file = [chunk.png_bytes for chunk in render_recording(path, self.config)]
        samples, _ = soundfile.read(path, dtype="float32")
        from_array = [
            chunk.png_bytes
            for chunk in render_waveform(np.asarray(samples, dtype=np.float32), self.config)
        ]

        self.assertEqual(from_file, from_array)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m unittest tests.preprocessing.test_recording -v`
Expected: FAIL at import with `ModuleNotFoundError: No module named 'frog_classifier.preprocessing.recording'`.

- [ ] **Step 3: Write the implementation**

Create `src/frog_classifier/preprocessing/recording.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import librosa
import numpy as np

from frog_classifier.data.config import PreprocessingConfig

from .spectrogram import chunk_image, encode_png, mel_decibels, noise_floor


@dataclass(frozen=True)
class RenderedChunk:
    start_s: int
    png_bytes: bytes


def load_waveform(path: Path, config: PreprocessingConfig) -> np.ndarray:
    """Mono waveform at the configured sample rate."""
    waveform, _ = librosa.load(
        path,
        sr=config.audio.sample_rate_hz,
        mono=config.audio.mono,
    )
    return np.asarray(waveform, dtype=np.float32)


def render_waveform(
    waveform: np.ndarray,
    config: PreprocessingConfig,
) -> Iterator[RenderedChunk]:
    """Yield one rendered chunk per complete window, in order.

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
        image = chunk_image(
            _mel_decibels(chunk, config),
            floor,
            db_floor=config.normalization.db_floor,
            db_ceiling=config.normalization.db_ceiling,
        )
        yield RenderedChunk(
            start_s=index * config.audio.chunk_seconds,
            png_bytes=encode_png(image),
        )


def render_recording(
    path: Path,
    config: PreprocessingConfig,
) -> Iterator[RenderedChunk]:
    """Render every complete chunk of one recording file."""
    return render_waveform(load_waveform(path, config), config)


def _mel_decibels(waveform: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    spectrogram = config.spectrogram
    return mel_decibels(
        waveform,
        sample_rate_hz=config.audio.sample_rate_hz,
        n_mels=spectrogram.n_mels,
        fmin_hz=spectrogram.fmin_hz,
        fmax_hz=spectrogram.fmax_hz,
        power=spectrogram.power,
        n_fft=spectrogram.n_fft,
        hop_length=spectrogram.hop_length,
    )
```

Replace `src/frog_classifier/preprocessing/__init__.py` with:

```python
from .recording import (
    RenderedChunk,
    load_waveform,
    render_recording,
    render_waveform,
)
from .spectrogram import (
    chunk_image,
    decode_png,
    encode_png,
    mel_decibels,
    noise_floor,
)

__all__ = [
    "RenderedChunk",
    "chunk_image",
    "decode_png",
    "encode_png",
    "load_waveform",
    "mel_decibels",
    "noise_floor",
    "render_recording",
    "render_waveform",
]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.preprocessing.test_recording -v`
Expected: `Ran 4 tests` and `OK`.

- [ ] **Step 5: Run the whole suite and commit**

Run: `uv run python -m unittest discover -s tests`
Expected: all pass.

```bash
git add src/frog_classifier/preprocessing tests/preprocessing/test_recording.py
git diff --cached --check
git commit -m "feat: render recordings against their own noise floor"
```

---

### Task 4: Slicer over the package

**Files:**
- Modify: `scripts/slice_audio.py` (full rewrite)
- Modify: `tests/test_slice_audio.py` (full rewrite)

**Interfaces:**
- Consumes: `load_preprocessing_config` from `frog_classifier.data`; `render_recording` from Task 3.
- Produces: `build_parser(repo_root: Path) -> argparse.ArgumentParser` with only `--raw-root`, `--out-root`, `--out-subdir`, `--limit-files`.
- Produces: `process_file(config, raw_root, out_root, audio_path) -> int` returning the number of PNGs written.
- Produces: `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write the failing tests**

Replace the entire content of `tests/test_slice_audio.py` with:

```python
from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from frog_classifier.preprocessing.spectrogram import decode_png
from tests.preprocessing.helpers import silence, tone, write_wav


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "slice_audio.py"


class SliceAudioTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw_root = self.root / "raw"
        self.out_root = self.root / "processed"
        self.module = self._load_module()

    def test_parser_exposes_no_preprocessing_overrides(self) -> None:
        parser = self.module.build_parser(REPO_ROOT)
        arguments = parser.parse_args([])

        for name in ("sample_rate", "chunk_seconds", "n_mels", "fmin", "fmax"):
            with self.subTest(option=name):
                self.assertFalse(hasattr(arguments, name))
        self.assertFalse(hasattr(self.module, "Config"))

    def test_writes_one_image_per_complete_chunk_preserving_folders(self) -> None:
        write_wav(
            self.raw_root / "pond" / "recording.wav",
            np.concatenate([silence(5), tone(1000.0, 7)]),
        )

        result, _ = self._run("--raw-root", str(self.raw_root), "--out-root", str(self.out_root))

        self.assertEqual(result, 0)
        written = sorted(path.name for path in (self.out_root / "pond").glob("*.png"))
        self.assertEqual(written, ["recording_start0s.png", "recording_start5s.png"])
        image = decode_png((self.out_root / "pond" / "recording_start5s.png").read_bytes())
        self.assertEqual(image.shape, (128, 216))

    def test_overwrites_an_existing_image_by_name(self) -> None:
        write_wav(self.raw_root / "pond" / "recording.wav", tone(1000.0, 5))
        stale = self.out_root / "pond" / "recording_start0s.png"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"stale")

        result, _ = self._run("--raw-root", str(self.raw_root), "--out-root", str(self.out_root))

        self.assertEqual(result, 0)
        self.assertNotEqual(stale.read_bytes(), b"stale")
        self.assertEqual(decode_png(stale.read_bytes()).shape, (128, 216))

    def test_continues_after_an_unreadable_file(self) -> None:
        broken = self.raw_root / "pond" / "broken.wav"
        broken.parent.mkdir(parents=True)
        broken.write_bytes(b"not audio")
        write_wav(self.raw_root / "pond" / "recording.wav", tone(1000.0, 5))

        result, output = self._run(
            "--raw-root", str(self.raw_root), "--out-root", str(self.out_root),
        )

        self.assertEqual(result, 0)
        self.assertIn("[ERROR]", output)
        self.assertIn("broken.wav", output)
        self.assertTrue((self.out_root / "pond" / "recording_start0s.png").is_file())

    def test_limit_files_bounds_each_pond(self) -> None:
        write_wav(self.raw_root / "pond" / "a.wav", tone(1000.0, 5))
        write_wav(self.raw_root / "pond" / "b.wav", tone(1000.0, 5))

        result, _ = self._run(
            "--raw-root", str(self.raw_root),
            "--out-root", str(self.out_root),
            "--limit-files", "1",
        )

        self.assertEqual(result, 0)
        written = sorted(path.name for path in (self.out_root / "pond").glob("*.png"))
        self.assertEqual(written, ["a_start0s.png"])

    def _run(self, *argv: str) -> tuple[int, str]:
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = self.module.main(argv)
        return result, output.getvalue()

    def _load_module(self):
        module_name = f"slice_audio_test_{id(self)}"
        specification = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)
        specification.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m unittest tests.test_slice_audio -v`
Expected: FAIL. `test_parser_exposes_no_preprocessing_overrides` fails because `build_parser` needs two arguments and the module still has `Config`; the others fail with `AttributeError` on `figure_width_inches`.

- [ ] **Step 3: Rewrite the slicer**

Replace the entire content of `scripts/slice_audio.py` with:

```python
from __future__ import annotations

"""
slice_audio.py

Slice every recording beneath a raw root into fixed five-second windows and
write one Mel-spectrogram PNG per complete window, preserving the folder
structure beneath the output root.

Every preprocessing value comes from config/preprocessing.toml through
frog_classifier.preprocessing. The command deliberately offers no overrides,
so stored images can never drift from the tracked contract.

Output naming: <stem>_start<N>s.png, where N is the window start in seconds.
Existing images with the same name are overwritten. Nothing is deleted.
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from tqdm import tqdm

from frog_classifier.data import PreprocessingConfig, load_preprocessing_config
from frog_classifier.preprocessing import render_recording


SUPPORTED_EXTS = {".wav", ".mp3"}


def iter_audio_files_os_walk(raw_root: Path) -> Iterable[Path]:
    """Find every supported audio file beneath raw_root."""
    for dirpath, _, filenames in os.walk(raw_root):
        directory = Path(dirpath)
        for name in filenames:
            path = directory / name
            if path.suffix.lower() in SUPPORTED_EXTS:
                yield path


def pond_name_for_path(raw_root: Path, audio_path: Path) -> str:
    """The top-level folder beneath raw_root, used to group progress."""
    relative = audio_path.relative_to(raw_root)
    return relative.parts[0] if len(relative.parts) > 0 else "raw"


def process_file(
    config: PreprocessingConfig,
    raw_root: Path,
    out_root: Path,
    audio_path: Path,
) -> int:
    """Render one recording and write its chunk images. Returns the count written."""
    relative = audio_path.relative_to(raw_root)
    out_dir = out_root / relative.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for chunk in render_recording(audio_path, config):
        out_path = out_dir / f"{audio_path.stem}_start{chunk.start_s}s.png"
        out_path.write_bytes(chunk.png_bytes)
        written += 1
    return written


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Slice raw audio into fixed windows and save Mel-spectrogram PNGs "
            "rendered against each recording's noise floor."
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=repo_root / "raw" / "external",
        help='Input root containing nested audio files (default: "raw/external").',
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=repo_root / "processed" / "external" / "spectrograms",
        help='Output root for spectrogram PNGs (default: "processed/external/spectrograms").',
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="",
        help="Optional subfolder under out-root.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="Process only the first N audio files per pond (0 = no limit).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_preprocessing_config(repo_root / "config" / "preprocessing.toml")
    args = build_parser(repo_root).parse_args(argv)

    raw_root = _resolve_path(repo_root, args.raw_root)
    out_root = _resolve_path(repo_root, args.out_root) / (args.out_subdir or "")
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")

    pond_to_files: dict[str, list[Path]] = defaultdict(list)
    for path in iter_audio_files_os_walk(raw_root):
        pond_to_files[pond_name_for_path(raw_root, path)].append(path)

    if not pond_to_files:
        print(f"No audio files found under {raw_root} (supported: {sorted(SUPPORTED_EXTS)})")
        return 0

    total_written = 0
    for pond in sorted(pond_to_files):
        files = sorted(pond_to_files[pond])
        if args.limit_files > 0:
            files = files[: args.limit_files]
        bar = tqdm(files, desc=f"Processing pond {pond}", unit="file")
        for audio_path in bar:
            try:
                total_written += process_file(config, raw_root, out_root, audio_path)
            except Exception as error:
                # Keep the run going; one unreadable file must not lose the batch.
                bar.write(f"[ERROR] {audio_path}: {error}")

    print(f"Done. Wrote {total_written} spectrogram PNG(s) to {out_root}")
    return 0


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_slice_audio -v`
Expected: `Ran 5 tests` and `OK`. The unreadable-file test prints a librosa or soundfile error line; that is the expected `[ERROR]` output.

- [ ] **Step 5: Run the whole suite and commit**

Run: `uv run python -m unittest discover -s tests`
Expected: all pass.

```bash
git add scripts/slice_audio.py tests/test_slice_audio.py
git diff --cached --check
git commit -m "feat: slice recordings through the preprocessing package"
```

---

### Task 5: Sync labeled images

**Files:**
- Create: `scripts/sync_labeled_images.py`
- Create: `tests/test_sync_labeled_images.py`

**Interfaces:**
- Consumes: `parse_example_filename`, `ExampleNameError`, `load_preprocessing_config` from `frog_classifier.data`.
- Produces: `SyncPlan(replacements: tuple[tuple[Path, Path], ...], issues: tuple[str, ...])` where each replacement is `(fresh_path, labeled_path)`.
- Produces: `plan_sync(labeled_root: Path, spectrogram_root: Path, *, chunk_seconds: int) -> SyncPlan`.
- Produces: `apply_sync(plan: SyncPlan) -> int`.
- Produces: `main(argv) -> int` with `--labeled-root`, `--spectrogram-root`, `--dry-run`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sync_labeled_images.py`:

```python
from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "sync_labeled_images.py"


class SyncLabeledImagesTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.labeled_root = self.root / "labeled"
        self.spectrogram_root = self.root / "processed" / "spectrograms"
        self.module = self._load_module()
        self.labeled = self._write(self.labeled_root / "litoria_aurea" / "rec_start5s.png", b"old")
        self.fresh = self._write(self.spectrogram_root / "site" / "rec_start5s.png", b"new")
        self.queued = self._write(self.spectrogram_root / "site" / "rec_start10s.png", b"queue")

    def test_replaces_labeled_bytes_and_removes_the_queue_copy(self) -> None:
        result, output = self._run()

        self.assertEqual(result, 0)
        self.assertIn("Replaced 1 labeled image", output)
        self.assertEqual(self.labeled.read_bytes(), b"new")
        self.assertFalse(self.fresh.exists())
        self.assertEqual(self.queued.read_bytes(), b"queue")

    def test_dry_run_changes_nothing(self) -> None:
        result, output = self._run("--dry-run")

        self.assertEqual(result, 0)
        self.assertIn("Dry run", output)
        self.assertEqual(self.labeled.read_bytes(), b"old")
        self.assertTrue(self.fresh.exists())

    def test_missing_counterpart_fails_without_touching_anything(self) -> None:
        self._write(self.labeled_root / "non_target" / "other_start0s.png", b"old")

        result, output = self._run()

        self.assertEqual(result, 2)
        self.assertIn("[missing_counterpart]", output)
        self.assertIn("other_start0s.png", output)
        self.assertEqual(self.labeled.read_bytes(), b"old")
        self.assertTrue(self.fresh.exists())

    def test_ambiguous_counterpart_fails_without_touching_anything(self) -> None:
        self._write(self.spectrogram_root / "elsewhere" / "rec_start5s.png", b"duplicate")

        result, output = self._run()

        self.assertEqual(result, 2)
        self.assertIn("[ambiguous_counterpart]", output)
        self.assertEqual(self.labeled.read_bytes(), b"old")
        self.assertTrue(self.fresh.exists())

    def test_unparseable_labeled_name_fails_without_touching_anything(self) -> None:
        self._write(self.labeled_root / "non_target" / "notes.png", b"old")

        result, output = self._run()

        self.assertEqual(result, 2)
        self.assertIn("[invalid_example_filename]", output)
        self.assertEqual(self.labeled.read_bytes(), b"old")
        self.assertTrue(self.fresh.exists())

    def test_nested_roots_are_rejected(self) -> None:
        result, output = self._run(
            "--labeled-root", str(self.spectrogram_root / "site"),
            "--spectrogram-root", str(self.spectrogram_root),
        )

        self.assertEqual(result, 2)
        self.assertIn("[nested_roots]", output)

    def _run(self, *extra: str) -> tuple[int, str]:
        argv = list(extra)
        if "--labeled-root" not in argv:
            argv += ["--labeled-root", str(self.labeled_root)]
        if "--spectrogram-root" not in argv:
            argv += ["--spectrogram-root", str(self.spectrogram_root)]
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = self.module.main(argv)
        return result, output.getvalue()

    def _write(self, path: Path, content: bytes) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _load_module(self):
        module_name = f"sync_labeled_images_test_{id(self)}"
        specification = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)
        specification.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m unittest tests.test_sync_labeled_images -v`
Expected: ERROR in `setUp` because `scripts/sync_labeled_images.py` does not exist.

- [ ] **Step 3: Write the script**

Create `scripts/sync_labeled_images.py`:

```python
from __future__ import annotations

"""
sync_labeled_images.py

After spectrograms are regenerated, every labeled example has a fresh copy
in the queue with the same file name. This command moves each fresh copy
over its labeled counterpart so the label keeps its identity and the queue
no longer contains a labeled example.

All checks run before any file is touched. Any issue aborts the command
with every issue listed and nothing changed.
"""

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from frog_classifier.data import (
    ExampleNameError,
    load_preprocessing_config,
    parse_example_filename,
)


@dataclass(frozen=True)
class SyncPlan:
    replacements: tuple[tuple[Path, Path], ...]
    issues: tuple[str, ...]


def plan_sync(
    labeled_root: Path,
    spectrogram_root: Path,
    *,
    chunk_seconds: int,
) -> SyncPlan:
    labeled_root = labeled_root.resolve()
    spectrogram_root = spectrogram_root.resolve()
    if _is_within(labeled_root, spectrogram_root) or _is_within(spectrogram_root, labeled_root):
        return SyncPlan((), (
            f"[nested_roots] {labeled_root} and {spectrogram_root} must not contain each other",
        ))
    if not labeled_root.is_dir():
        return SyncPlan((), (f"[missing_root] labeled root is not a directory: {labeled_root}",))
    if not spectrogram_root.is_dir():
        return SyncPlan((), (
            f"[missing_root] spectrogram root is not a directory: {spectrogram_root}",
        ))

    fresh_by_name: dict[str, list[Path]] = defaultdict(list)
    for path in spectrogram_root.rglob("*.png"):
        if path.is_file():
            fresh_by_name[path.name].append(path)

    replacements: list[tuple[Path, Path]] = []
    issues: list[str] = []
    for labeled in sorted(labeled_root.rglob("*.png")):
        if not labeled.is_file():
            continue
        try:
            parse_example_filename(labeled, chunk_seconds=chunk_seconds)
        except ExampleNameError as error:
            issues.append(f"[invalid_example_filename] {labeled}: {error}")
            continue
        candidates = sorted(fresh_by_name.get(labeled.name, []))
        if not candidates:
            issues.append(
                f"[missing_counterpart] {labeled}: no image named {labeled.name} "
                f"under {spectrogram_root}"
            )
        elif len(candidates) > 1:
            issues.append(
                f"[ambiguous_counterpart] {labeled}: {len(candidates)} images named "
                f"{labeled.name} under {spectrogram_root}"
            )
        else:
            replacements.append((candidates[0], labeled))
    return SyncPlan(tuple(replacements), tuple(issues))


def apply_sync(plan: SyncPlan) -> int:
    """Move each fresh image over its labeled counterpart. Returns the count moved."""
    for fresh, labeled in plan.replacements:
        os.replace(fresh, labeled)
    return len(plan.replacements)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replace labeled images with their freshly regenerated counterparts.",
    )
    parser.add_argument("--labeled-root", type=Path, default=Path("labeled"))
    parser.add_argument(
        "--spectrogram-root",
        type=Path,
        default=Path("processed/external/spectrograms"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    args = build_parser().parse_args(argv)
    config = load_preprocessing_config(repo_root / "config" / "preprocessing.toml")
    plan = plan_sync(
        _resolve(repo_root, args.labeled_root),
        _resolve(repo_root, args.spectrogram_root),
        chunk_seconds=config.audio.chunk_seconds,
    )
    if plan.issues:
        print("Labeled image sync failed; no files were changed:", file=sys.stderr)
        for issue in plan.issues:
            print(issue, file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"Dry run: {len(plan.replacements)} labeled image(s) would be replaced")
        return 0
    replaced = apply_sync(plan)
    print(f"Replaced {replaced} labeled image(s) from {args.spectrogram_root}")
    return 0


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_sync_labeled_images -v`
Expected: `Ran 6 tests` and `OK`.

- [ ] **Step 5: Run the whole suite and commit**

Run: `uv run python -m unittest discover -s tests`
Expected: all pass.

```bash
git add scripts/sync_labeled_images.py tests/test_sync_labeled_images.py
git diff --cached --check
git commit -m "feat: replace labeled images from regenerated spectrograms by name"
```

---

### Task 6: Verify stored spectrograms against raw audio

**Files:**
- Create: `scripts/verify_spectrograms.py`
- Create: `tests/test_verify_spectrograms.py`

**Interfaces:**
- Consumes: `render_recording` from Task 3; `parse_example_filename`, `ExampleNameError`, `load_preprocessing_config` from `frog_classifier.data`.
- Produces: `VerificationResult(image_path: Path, status: str)` with status in `MATCH`, `DIFFERENT`, `MISSING_RECORDING`, `MISSING_CHUNK`, `INVALID_NAME`.
- Produces: `find_recording(raw_root: Path, stem: str) -> Path | None`.
- Produces: `verify_images(image_paths, raw_root, config) -> list[VerificationResult]`.
- Produces: `main(argv) -> int` with `--raw-root`, `--spectrogram-root`, `--labeled-root`, `--sample`, `--seed`, `--all`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_verify_spectrograms.py`:

```python
from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from frog_classifier.data import load_preprocessing_config
from frog_classifier.preprocessing import render_recording
from tests.preprocessing.helpers import silence, tone, write_wav


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_spectrograms.py"


class VerifySpectrogramsTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw_root = self.root / "raw"
        self.spectrogram_root = self.root / "processed" / "spectrograms"
        self.labeled_root = self.root / "labeled"
        self.module = self._load_module()
        config = load_preprocessing_config(REPO_ROOT / "config" / "preprocessing.toml")
        recording = write_wav(
            self.raw_root / "site" / "rec.wav",
            np.concatenate([silence(5), tone(1000.0, 5), silence(2)]),
        )
        chunks = {chunk.start_s: chunk.png_bytes for chunk in render_recording(recording, config)}
        self.queued = self._write(self.spectrogram_root / "site" / "rec_start0s.png", chunks[0])
        self.labeled = self._write(self.labeled_root / "litoria_aurea" / "rec_start5s.png", chunks[5])

    def test_consistent_tree_matches_everywhere(self) -> None:
        result, output = self._run("--all")

        self.assertEqual(result, 0)
        self.assertIn("MATCH", output)
        self.assertNotIn("DIFFERENT", output)
        self.assertIn("2 checked, 2 matched", output)

    def test_tampered_image_is_reported_and_fails(self) -> None:
        self.labeled.write_bytes(self.labeled.read_bytes() + b"\x00")

        result, output = self._run("--all")

        self.assertEqual(result, 2)
        self.assertIn(f"DIFFERENT {self.labeled}", output)

    def test_missing_recording_is_reported_and_fails(self) -> None:
        self._write(self.labeled_root / "non_target" / "ghost_start0s.png", b"x")

        result, output = self._run("--all")

        self.assertEqual(result, 2)
        self.assertIn("MISSING_RECORDING", output)
        self.assertIn("ghost_start0s.png", output)

    def test_chunk_beyond_recording_is_reported_and_fails(self) -> None:
        self._write(self.spectrogram_root / "site" / "rec_start30s.png", b"x")

        result, output = self._run("--all")

        self.assertEqual(result, 2)
        self.assertIn("MISSING_CHUNK", output)

    def test_sample_is_seeded_and_bounded(self) -> None:
        result, output = self._run("--sample", "1", "--seed", "3")

        self.assertEqual(result, 0)
        self.assertIn("2 checked, 2 matched", output)

    def _run(self, *extra: str) -> tuple[int, str]:
        argv = [
            "--raw-root", str(self.raw_root),
            "--spectrogram-root", str(self.spectrogram_root),
            "--labeled-root", str(self.labeled_root),
            *extra,
        ]
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = self.module.main(argv)
        return result, output.getvalue()

    def _write(self, path: Path, content: bytes) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _load_module(self):
        module_name = f"verify_spectrograms_test_{id(self)}"
        specification = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)
        specification.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
```

Note on `test_sample_is_seeded_and_bounded`: `--sample 1` draws one image from the labeled tree and one from the queue, so two are checked.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m unittest tests.test_verify_spectrograms -v`
Expected: ERROR in `setUp` because `scripts/verify_spectrograms.py` does not exist.

- [ ] **Step 3: Write the script**

Create `scripts/verify_spectrograms.py`:

```python
from __future__ import annotations

"""
verify_spectrograms.py

Prove that stored spectrogram images were rendered under the tracked
preprocessing contract by re-rendering their recordings from raw audio and
comparing bytes. Samples from both the labeled tree and the queue, or checks
everything with --all.
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from frog_classifier.data import (
    ExampleNameError,
    PreprocessingConfig,
    load_preprocessing_config,
    parse_example_filename,
)
from frog_classifier.preprocessing import render_recording


SUPPORTED_EXTS = {".wav", ".mp3"}


@dataclass(frozen=True)
class VerificationResult:
    image_path: Path
    status: str


def find_recording(raw_root: Path, stem: str) -> Path | None:
    """The single recording with this stem beneath raw_root, or None."""
    matches = sorted(
        path
        for path in raw_root.rglob(f"{stem}.*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
    )
    return matches[0] if len(matches) == 1 else None


def verify_images(
    image_paths: Iterable[Path],
    raw_root: Path,
    config: PreprocessingConfig,
) -> list[VerificationResult]:
    results: list[VerificationResult] = []
    by_stem: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for path in image_paths:
        try:
            key = parse_example_filename(path, chunk_seconds=config.audio.chunk_seconds)
        except ExampleNameError:
            results.append(VerificationResult(path, "INVALID_NAME"))
            continue
        by_stem[key.recording_id].append((path, key.start_s))

    for stem, items in sorted(by_stem.items()):
        recording = find_recording(raw_root, stem)
        if recording is None:
            results.extend(VerificationResult(path, "MISSING_RECORDING") for path, _ in items)
            continue
        rendered = {chunk.start_s: chunk.png_bytes for chunk in render_recording(recording, config)}
        for path, start_s in items:
            expected = rendered.get(start_s)
            if expected is None:
                status = "MISSING_CHUNK"
            elif path.read_bytes() == expected:
                status = "MATCH"
            else:
                status = "DIFFERENT"
            results.append(VerificationResult(path, status))
    return sorted(results, key=lambda result: str(result.image_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-render sampled spectrograms from raw audio and compare bytes.",
    )
    parser.add_argument("--raw-root", type=Path, default=Path("raw/external"))
    parser.add_argument(
        "--spectrogram-root",
        type=Path,
        default=Path("processed/external/spectrograms"),
    )
    parser.add_argument("--labeled-root", type=Path, default=Path("labeled"))
    parser.add_argument("--sample", type=int, default=24, help="Images per tree to check.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--all", action="store_true", help="Check every image in both trees.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    args = build_parser().parse_args(argv)
    config = load_preprocessing_config(repo_root / "config" / "preprocessing.toml")
    raw_root = _resolve(repo_root, args.raw_root)
    trees = (
        _resolve(repo_root, args.labeled_root),
        _resolve(repo_root, args.spectrogram_root),
    )

    selected: list[Path] = []
    rng = random.Random(args.seed)
    for tree in trees:
        images = sorted(path for path in tree.rglob("*.png") if path.is_file())
        if args.all:
            selected.extend(images)
        else:
            selected.extend(rng.sample(images, min(args.sample, len(images))))

    if not selected:
        print("No images found to verify", file=sys.stderr)
        return 2

    results = verify_images(selected, raw_root, config)
    for result in results:
        print(f"{result.status} {result.image_path}")
    counts = Counter(result.status for result in results)
    print(f"{len(results)} checked, {counts.get('MATCH', 0)} matched")
    for status in sorted(status for status in counts if status != "MATCH"):
        print(f"{counts[status]} {status}", file=sys.stderr)
    return 0 if counts.get("MATCH", 0) == len(results) else 2


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_verify_spectrograms -v`
Expected: `Ran 5 tests` and `OK`.

- [ ] **Step 5: Run the whole suite and commit**

Run: `uv run python -m unittest discover -s tests`
Expected: all pass.

```bash
git add scripts/verify_spectrograms.py tests/test_verify_spectrograms.py
git diff --cached --check
git commit -m "feat: verify stored spectrograms by re-rendering from raw audio"
```

---

### Task 7: Labeler display through a colour map

**Files:**
- Create: `src/frog_classifier/preprocessing/display.py`
- Modify: `src/frog_classifier/preprocessing/__init__.py`
- Create: `tests/preprocessing/test_display.py`
- Modify: `scripts/label_frontend.py` (the `st.image` call and imports)
- Modify: `scripts/label_spectrograms.py` (`_init_image_window` and `_update_image_window`)

**Interfaces:**
- Consumes: `encode_png` from Task 1 in the test.
- Produces: `colorize(png_bytes: bytes) -> np.ndarray` of shape `(height, width, 3)` and dtype `uint8`, viridis mapped.

- [ ] **Step 1: Write the failing test**

Create `tests/preprocessing/test_display.py`:

```python
from __future__ import annotations

import unittest

import numpy as np

from frog_classifier.preprocessing.display import colorize
from frog_classifier.preprocessing.spectrogram import encode_png


class ColorizeTests(unittest.TestCase):
    def test_maps_grayscale_through_viridis(self) -> None:
        image = np.array([[0, 255], [128, 64]], dtype=np.uint8)

        rgb = colorize(encode_png(image))

        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(rgb[0, 0].tolist(), [68, 1, 84])
        self.assertEqual(rgb[0, 1].tolist(), [253, 231, 37])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python -m unittest tests.preprocessing.test_display -v`
Expected: FAIL at import with `ModuleNotFoundError: No module named 'frog_classifier.preprocessing.display'`.

- [ ] **Step 3: Write the implementation**

Create `src/frog_classifier/preprocessing/display.py`:

```python
from __future__ import annotations

import io

import matplotlib
import numpy as np
from PIL import Image


def colorize(png_bytes: bytes) -> np.ndarray:
    """RGB uint8 view of a grayscale PNG through the viridis colour map.

    Storage stays grayscale; this exists only so the labelers show the
    familiar colours.
    """
    with Image.open(io.BytesIO(png_bytes)) as image:
        grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    rgba = matplotlib.colormaps["viridis"](grayscale)
    return np.rint(rgba[:, :, :3] * 255.0).astype(np.uint8)
```

In `src/frog_classifier/preprocessing/__init__.py` add `from .display import colorize` as the first import line and `"colorize",` to `__all__` after `"chunk_image",`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run python -m unittest tests.preprocessing.test_display -v`
Expected: `Ran 1 test` and `OK`.

- [ ] **Step 5: Update the Streamlit labeler**

In `scripts/label_frontend.py`, after the line `import streamlit as st`, add:

```python
from frog_classifier.preprocessing import colorize
```

Replace:

```python
    with col_img:
        st.image(img_bytes, use_container_width=True)
```

with:

```python
    with col_img:
        st.image(colorize(img_bytes), use_container_width=True)
```

- [ ] **Step 6: Update the Matplotlib labeler**

In `scripts/label_spectrograms.py`, inside `_init_image_window`, replace:

```python
    with Image.open(img_path) as img:
        img_arr = np.asarray(img.convert("RGB"))

    fig, ax = plt.subplots()
    # Leave room for buttons at the bottom.
    fig.subplots_adjust(bottom=0.18)
    im = ax.imshow(img_arr)
```

with:

```python
    with Image.open(img_path) as img:
        img_arr = np.asarray(img.convert("L"))

    fig, ax = plt.subplots()
    # Leave room for buttons at the bottom.
    fig.subplots_adjust(bottom=0.18)
    im = ax.imshow(img_arr, cmap="viridis", vmin=0, vmax=255, aspect="auto")
```

Inside `_update_image_window`, replace:

```python
    with Image.open(img_path) as img:
        img_arr = np.asarray(img.convert("RGB"))
    im.set_data(img_arr)
```

with:

```python
    with Image.open(img_path) as img:
        img_arr = np.asarray(img.convert("L"))
    im.set_data(img_arr)
```

- [ ] **Step 7: Compile the scripts and inspect the Streamlit labeler by hand**

Run: `uv run python -B -m py_compile scripts/label_frontend.py scripts/label_spectrograms.py`
Expected: no output.

Create two synthetic images to look at, without touching project data:

```bash
uv run python -c "
from pathlib import Path
import numpy as np
from frog_classifier.data import load_preprocessing_config
from frog_classifier.preprocessing import render_waveform
from tests.preprocessing.helpers import silence, tone
config = load_preprocessing_config(Path('config/preprocessing.toml'))
wave = np.concatenate([silence(5), tone(1000.0, 5)])
out = Path('.scratch/spectrograms/demo'); out.mkdir(parents=True, exist_ok=True)
for chunk in render_waveform(wave, config):
    (out / f'demo_start{chunk.start_s}s.png').write_bytes(chunk.png_bytes)
print(sorted(p.name for p in out.glob('*.png')))
"
uv run python -m streamlit run scripts/label_frontend.py --server.address localhost --server.port 8501
```

In the browser, open Advanced settings, set `spectrogram_root` to the absolute path of `.scratch/spectrograms` and `training_root` to `.scratch/labeled`, click Reload images, and confirm the second image shows a bright horizontal band in viridis colours near the lower third of the picture. Stop Streamlit. Delete `.scratch/` afterwards; it is not part of the repository and must never be committed.

- [ ] **Step 8: Run the whole suite and commit**

Run: `uv run python -m unittest discover -s tests`
Expected: all pass.

```bash
git add src/frog_classifier/preprocessing/display.py src/frog_classifier/preprocessing/__init__.py tests/preprocessing/test_display.py scripts/label_frontend.py scripts/label_spectrograms.py
git diff --cached --check
git commit -m "feat: display grayscale spectrograms through viridis in both labelers"
```

---

### Task 8: Repository checks, provenance note, and documentation

**Files:**
- Modify: `tests/test_repository_reproducibility.py`
- Modify: `src/frog_classifier/data/reporting.py`
- Modify: `README.md`, `docs/outline.md`, `docs/architecture.md`, `docs/workflow.md`

- [ ] **Step 1: Extend the reproducibility test so it fails on the current docs**

In `tests/test_repository_reproducibility.py`, after the `DATA_PACKAGE_MODULES` tuple add:

```python
PREPROCESSING_PACKAGE_MODULES = (
    "src/frog_classifier/preprocessing/__init__.py",
    "src/frog_classifier/preprocessing/display.py",
    "src/frog_classifier/preprocessing/recording.py",
    "src/frog_classifier/preprocessing/spectrogram.py",
)
```

Replace the `OPERATIONAL_DOCUMENTS` tuple with:

```python
OPERATIONAL_DOCUMENTS = (
    "README.md",
    "roadmap.md",
    "docs/outline.md",
    "docs/architecture.md",
    "docs/workflow.md",
    "docs/decisions/2026-09-13-spectrogram-rendering.md",
)
```

Replace the `OPERATIONAL_TEXT_FILES` tuple with:

```python
OPERATIONAL_TEXT_FILES = (
    "README.md",
    "docs/outline.md",
    "docs/architecture.md",
    "docs/workflow.md",
    "scripts/build_manifest.py",
    "scripts/label_spectrograms.py",
    "scripts/slice_audio.py",
    "scripts/sync_labeled_images.py",
    "scripts/train_baseline.py",
    "scripts/verify_spectrograms.py",
)
```

In `test_project_navigation_files_are_complete_and_linked`, change the loop header to:

```python
        for relative_path in (
            *OPERATIONAL_DOCUMENTS,
            "config/preprocessing.toml",
            *DATA_PACKAGE_MODULES,
            *PREPROCESSING_PACKAGE_MODULES,
        ):
```

Add a new test method to `OperationalDocumentationTests`:

```python
    def test_workflow_documents_every_operational_command(self) -> None:
        workflow = (REPO_ROOT / "docs/workflow.md").read_text(encoding="utf-8")
        for command in (
            "uv run python scripts/slice_audio.py",
            "uv run python scripts/sync_labeled_images.py",
            "uv run python scripts/verify_spectrograms.py --sample 50",
            "uv run python scripts/build_manifest.py",
            "uv run python scripts/train_baseline.py",
        ):
            with self.subTest(command=command):
                self.assertIn(command, workflow)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python -m unittest tests.test_repository_reproducibility -v`
Expected: FAIL in `test_workflow_documents_every_operational_command` because the sync and verify commands are not yet in the workflow.

- [ ] **Step 3: Update the provenance note**

In `src/frog_classifier/data/reporting.py` replace:

```python
PROVENANCE_NOTE = (
    "The preprocessing checksum records the declared configuration but does "
    "not cryptographically prove how pre-existing PNG files were generated."
)
```

with:

```python
PROVENANCE_NOTE = (
    "The preprocessing checksum records the declared configuration. Run "
    "scripts/verify_spectrograms.py to prove stored images were rendered under it."
)
```

- [ ] **Step 4: Update the README**

In `README.md`, after the sentence that begins `Frog Classifier turns long field recordings into fixed five-second Mel-spectrogram examples`, add on its own line:

```markdown
Each image shows decibels above its own recording's noise floor, so faint calls stay visible and brightness means the same thing in every image.
```

After the "Process pond recordings separately" code block and before "Launch the terminal labeler", add:

````markdown
Replace the labeled images after regenerating spectrograms:

```powershell
uv run python scripts/sync_labeled_images.py
```

Prove a random sample of stored spectrograms matches the raw audio and the tracked contract:

```powershell
uv run python scripts/verify_spectrograms.py --sample 50
```
````

- [ ] **Step 5: Update the outline**

In `docs/outline.md`, in the "2. Spectrogram generation" section, add after the existing two sentences:

```markdown
Each image shows decibels above the recording's own noise floor per frequency band, which keeps faint calls visible and removes steady background such as insect drone.
```

In the repository structure block, add these lines in the positions shown:

```text
  scripts/                     commands a person runs
    slice_audio.py             generate spectrograms
    sync_labeled_images.py     replace labeled images after regeneration
    verify_spectrograms.py     prove stored images match the contract
    label_frontend.py          Streamlit labeling interface
    label_spectrograms.py      Matplotlib labeling interface
    build_manifest.py          validate labels and build reports
    train_baseline.py          run the strict sanity baseline

  src/frog_classifier/         reusable Python package
    data/                      config, naming, manifests, folds, validation, and reports
    preprocessing/             Mel rendering, noise floor, PNG encoding, and display colours
```

and under `docs/`:

```text
    decisions/                 dated design decisions with their evidence
```

In the "Where to make a change" table add a row after the "Audio and spectrogram defaults" row:

```markdown
| Spectrogram rendering behavior | `src/frog_classifier/preprocessing/` |
```

- [ ] **Step 6: Update the architecture document**

In `docs/architecture.md`, in the storage contract block, add after the `src/frog_classifier/data/` line:

```text
src/frog_classifier/preprocessing/   Mel rendering, per-recording noise floor, PNG encoding, display colours
docs/decisions/              dated design decisions with their evidence
```

Replace the whole "Preprocessing contract" section with:

```markdown
## Preprocessing contract

[`config/preprocessing.toml`](../config/preprocessing.toml) is the machine-readable source of truth for the fixed-window audio, Mel-spectrogram, normalization, rendering, and canonical class settings.
Schema version 2 declares 22,050 Hz mono audio, non-overlapping five-second windows, dropped incomplete final chunks, 128 Mel bins spanning 400 Hz to 4,000 Hz, a per-recording noise floor at the 50th percentile, a 0 dB to 30 dB display range, 8-bit grayscale PNG output with the lowest band at the bottom, and the `litoria_aurea: 1` and `non_target: 0` mapping.
The loader accepts only schema version 2 and tells the operator to regenerate spectrograms when it meets an earlier schema.
The slicer takes every preprocessing value from this file and offers no command-line overrides.
The data package reads the exact TOML bytes and includes their SHA-256 checksum in each generated manifest row.
It rejects non-finite floating-point values before preprocessing begins.

The band was corrected on September 13, 2026 after regeneration from raw audio proved the existing spectrograms were produced with 400 Hz to 4,000 Hz rather than the previously declared 0 Hz to 8,000 Hz.
The noise-floor rendering was adopted the same day; the comparison and reasoning are in [the rendering decision](decisions/2026-09-13-spectrogram-rendering.md).

## Spectrogram rendering

`frog_classifier.preprocessing` renders one recording at a time.
It loads the whole file as mono audio at the configured rate, computes the Mel power spectrogram of the entire file in absolute decibels, and takes the configured percentile of every band across all frames as that band's noise floor.
Each complete five-second chunk is then converted from its own samples, the floor is subtracted per band, and decibels from `db_floor` to `db_ceiling` map linearly to grey levels 0 to 255 with clipping outside that range.
Rows are flipped so the lowest band is the bottom row, and the array is written as a lossless 8-bit grayscale PNG with no metadata, so identical audio always yields identical bytes.
Under the current contract every image is 216 pixels wide and 128 pixels tall.
The labelers show these images through the viridis colour map for readability; storage stays grayscale.

## Regeneration and verification

`scripts/slice_audio.py` overwrites images by name and never deletes.
`scripts/sync_labeled_images.py` moves each freshly rendered image over its labeled counterpart by name after all names are checked, so a label keeps its identity and the queue no longer holds a labeled example.
`scripts/verify_spectrograms.py` re-renders a seeded sample, or every image with `--all`, from raw audio and compares bytes, which proves the stored images match the tracked contract.
```

- [ ] **Step 7: Update the workflow document**

In `docs/workflow.md`, replace the "Generate spectrograms" section up to and including the sentence `For example, ...` with:

````markdown
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
Pass `--labeled-root` and `--spectrogram-root` together when syncing pond data.

## Verify stored spectrograms

Prove that stored images match the raw audio and the tracked contract:

```powershell
uv run python scripts/verify_spectrograms.py --sample 50
```

The command draws a seeded sample from the labeled tree and from the queue, re-renders each recording, and prints `MATCH` or the reason for a mismatch per image.
It exits non-zero on any mismatch.
Pass `--all` to check every image, which renders every recording once.
````

In the "Run the strict baseline and the complete repository verification" code block, replace the `py_compile` line with:

```powershell
uv run python -B -m py_compile scripts/build_manifest.py scripts/label_frontend.py scripts/label_spectrograms.py scripts/slice_audio.py scripts/sync_labeled_images.py scripts/train_baseline.py scripts/verify_spectrograms.py
```

- [ ] **Step 8: Run the checks**

Run: `uv run python -m unittest discover -s tests`
Expected: all pass, including the new workflow command test.

Run: `git diff --check`
Expected: no output.

Scan the changed Markdown for the em dash character:

```powershell
foreach ($path in @(git diff --name-only -- '*.md')) { Select-String -LiteralPath $path -Pattern ([char]0x2014) }
```

Expected: no output.

- [ ] **Step 9: Commit**

```bash
git add tests/test_repository_reproducibility.py src/frog_classifier/data/reporting.py README.md docs/outline.md docs/architecture.md docs/workflow.md
git diff --cached --check
git commit -m "docs: describe noise-floor rendering, sync, and verification commands"
```

---

### Task 9: Regenerate the real data (main checkout only)

This task runs in `D:\projects\frog-classifier` on the feature branch after Tasks 1 to 8 are merged into it or checked out there. It touches ignored data and must be done in order. Expect about twenty to thirty minutes of compute for the full slice.

**Files:**
- Modify: `roadmap.md`

- [ ] **Step 1: Confirm the starting state**

```powershell
git status --short
(Get-ChildItem -Recurse -Filter *.png processed/external/spectrograms | Measure-Object).Count
(Get-ChildItem -Recurse -Filter *.png labeled | Measure-Object).Count
```

Expected: clean tree, 37019, and 61. If the counts differ, stop and report before continuing.

- [ ] **Step 2: Smoke the slicer into a scratch folder**

```powershell
uv run python scripts/slice_audio.py --limit-files 1 --out-root .scratch/smoke
Get-ChildItem -Recurse -Filter *.png .scratch/smoke | Measure-Object
```

Expected: 8 recordings processed, one per top-level folder, 480 images written. Open two of them through the Streamlit labeler as in Task 7 Step 7 and confirm they render. Then remove `.scratch/`.

- [ ] **Step 3: Regenerate the whole queue**

```powershell
uv run python scripts/slice_audio.py
```

Expected final line: `Done. Wrote 37080 spectrogram PNG(s) to D:\projects\frog-classifier\processed\external\spectrograms`.

- [ ] **Step 4: Replace the labeled images**

```powershell
uv run python scripts/sync_labeled_images.py --dry-run
uv run python scripts/sync_labeled_images.py
(Get-ChildItem -Recurse -Filter *.png processed/external/spectrograms | Measure-Object).Count
(Get-ChildItem -Recurse -Filter *.png labeled | Measure-Object).Count
```

Expected: `Dry run: 61 labeled image(s) would be replaced`, then `Replaced 61 labeled image(s)`, then 37019 and 61.

- [ ] **Step 5: Verify a sample against raw audio**

```powershell
uv run python scripts/verify_spectrograms.py --sample 50
```

Expected: every line begins with `MATCH`, the summary reads `100 checked, 100 matched`, and the exit code is 0.

- [ ] **Step 6: Rebuild the manifest twice and run the baseline**

```powershell
uv run python scripts/build_manifest.py
Get-FileHash -Algorithm SHA256 -LiteralPath labeled/manifest.csv, results/data_quality/manifest-report.json, results/data_quality/manifest-report.md
uv run python scripts/build_manifest.py
Get-FileHash -Algorithm SHA256 -LiteralPath labeled/manifest.csv, results/data_quality/manifest-report.json, results/data_quality/manifest-report.md
uv run python scripts/train_baseline.py
```

Expected: 61 examples from 60 recording groups, identical hashes across the two builds, and the baseline prints train, validation, and test metrics. Copy the three hashes for the roadmap.

- [ ] **Step 7: Full repository verification**

```powershell
uv lock --check
uv sync --frozen
uv pip check
uv run python -m unittest discover -s tests -v
uv run python -B -m py_compile scripts/build_manifest.py scripts/label_frontend.py scripts/label_spectrograms.py scripts/slice_audio.py scripts/sync_labeled_images.py scripts/train_baseline.py scripts/verify_spectrograms.py
git diff --check
git status --short
```

Expected: all pass, no output from the diff check, and `git status --short` lists only `roadmap.md` after the next step.

- [ ] **Step 8: Record the checkpoint in the roadmap**

In `roadmap.md`, replace the "Current checkpoint" list header line `Verified on July 22, 2026:` with `Verified on <today's date>:`, change `- Active Git branch: \`modern\`` to `- Active Git branch: \`main\``, and add after the list:

```markdown
Every spectrogram was regenerated under preprocessing schema version 2 with per-recording noise-floor rendering, and a 100-image sample re-rendered from raw audio byte for byte.
```

Extend the "Corrected on September 13, 2026" list at the end of the Phase 2 verified acceptance with:

```markdown
- Spectrogram rendering moved to decibels above each recording's noise floor under preprocessing schema version 2; see the [rendering decision](docs/decisions/2026-09-13-spectrogram-rendering.md).
- The regenerated manifest and reports have the SHA-256 values `<csv hash>`, `<json hash>`, and `<markdown hash>`, which supersede the July 22 values above.
```

Fill in the three hashes from Step 6.

- [ ] **Step 9: Commit**

```bash
git add roadmap.md
git diff --cached --check
git commit -m "docs: record the regenerated spectrogram checkpoint"
```

---

## Self-review against the spec

- Package layout, pure functions, recording renderer, and display colour map: Tasks 1, 3, 7.
- Schema 2 loader, rejection messages, tracked configuration: Task 2.
- Slicer without overrides, overwrite by name, per-file error tolerance: Task 4.
- Sync command with all-checks-first behaviour, nested-root guard, dry run: Task 5.
- Verify command with seeded sample, `--all`, per-image status, non-zero exit: Task 6.
- Labeler display through viridis, storage grayscale: Task 7.
- Provenance note, reproducibility test coverage of new modules and scripts, documentation updates: Task 8.
- Migration runbook, counts, byte-identical sample, deterministic manifest, baseline, roadmap checkpoint: Task 9.
- Not deleting raw audio, playback chunks, or labels: no task deletes; Task 5 moves fresh queue copies only.
