# ResNet Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train, evaluate, save, and verify an ImageNet-pretrained ResNet-18 *Litoria aurea* detector from the validated manifest.

**Architecture:** A new `frog_classifier.training` package holds small modules for configuration, image loading, augmentation, the model, the training loop, metrics, the baseline, the saved run folder, the report, the run orchestration, and verification. Two thin commands, `scripts/train_model.py` and `scripts/verify_model.py`, wrap it. PyTorch arrives as a default uv dependency group so the labeler export stays unchanged.

**Tech Stack:** Python 3.13, PyTorch 2.14 and torchvision 0.29 from the CUDA 13.0 index, scikit-learn metrics, Pillow, matplotlib, unittest.

**Spec:** `docs/superpowers/specs/2026-10-14-resnet-training-design.md`

## Global Constraints

- Python stays `>=3.13,<3.14`; the existing runtime dependency list does not change.
- `torch>=2.14,<2.15` and `torchvision>=0.29,<0.30` live in the `model` dependency group, from the explicit `pytorch-cu130` index (`https://download.pytorch.org/whl/cu130`) on Windows and Linux.
- `requirements_labeler.txt` is generated with `uv export --frozen --no-default-groups --no-hashes --format requirements-txt --output-file requirements_labeler.txt` and contains no PyTorch.
- uv is at `/c/Users/ghimi/.local/bin/uv.exe`; it is not on PATH in this shell. Run Python as `.venv/Scripts/python.exe`.
- Text files use LF line endings; write files with the editor tools or bytes, never Python `write_text` on Windows.
- Never use the em dash character anywhere. Each full sentence in long Markdown files sits on its own line.
- Commit messages carry no co-author trailer.
- No test downloads weights or needs a GPU: test configurations set `pretrained = false` and pass a CPU device.
- Generated runs go under `models/`, which Git ignores. Never commit `labeled/`, `models/`, or `results/` contents.

## Review Focus

- A manifest built with a non-default test or validation fold must still make rotation zero match the manifest's own split; a run must not silently re-split.
- A clip relabeled or returned to the queue after training must make verification report a missing or changed image, not crash or report a wrong prediction.
- A machine without CUDA must train on the CPU by default, and an explicit CUDA request without a GPU must fail with a clear message.
- A repository without a decision log must still train, reporting zero faint clips.
- A run that fails partway must never leave a folder under its final name.

---

### Task 1: PyTorch dependency group

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` (generated)
- Modify: `requirements_labeler.txt` (generated)
- Test: `tests/test_repository_reproducibility.py`

**Interfaces:**
- Produces: an environment where `import torch, torchvision` works and `torch.cuda.is_available()` is true on this machine.

- [ ] **Step 1: Write the failing dependency test**

In `tests/test_repository_reproducibility.py`, add after `EXPECTED_RUNTIME_REQUIREMENTS`:

```python
MODEL_GROUP_REQUIREMENTS = {
    "torch>=2.14,<2.15",
    "torchvision>=0.29,<0.30",
}
```

Append to `test_dependency_metadata_and_compatibility_export_agree`, after the `exported_names` assertion:

```python
        self.assertEqual(set(pyproject["dependency-groups"]["model"]), MODEL_GROUP_REQUIREMENTS)
        self.assertEqual(pyproject["tool"]["uv"]["default-groups"], ["model"])
        indexes = {index["name"]: index for index in pyproject["tool"]["uv"]["index"]}
        self.assertEqual(indexes["pytorch-cu130"]["url"], "https://download.pytorch.org/whl/cu130")
        self.assertTrue(indexes["pytorch-cu130"]["explicit"])
        for package in ("torch", "torchvision"):
            with self.subTest(source=package):
                sources = pyproject["tool"]["uv"]["sources"][package]
                self.assertEqual({source["index"] for source in sources}, {"pytorch-cu130"})
        # The launcher installs this export with pip; PyTorch would add gigabytes it never uses.
        export_text = export_path.read_text(encoding="utf-8")
        self.assertIn("--no-default-groups", export_text)
        self.assertTrue({"torch", "torchvision"}.isdisjoint(exported_names))
```

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/Scripts/python.exe -m unittest tests.test_repository_reproducibility -v`
Expected: FAIL with `KeyError: 'dependency-groups'`.

- [ ] **Step 3: Declare the group and index**

In `pyproject.toml`, insert before `[build-system]`:

```toml
[dependency-groups]
model = [
    "torch>=2.14,<2.15",
    "torchvision>=0.29,<0.30",
]
```

Replace the `[tool.uv]` table with:

```toml
[tool.uv]
package = true
link-mode = "copy"
default-groups = ["model"]

[tool.uv.sources]
torch = [{ index = "pytorch-cu130", marker = "sys_platform == 'win32' or sys_platform == 'linux'" }]
torchvision = [{ index = "pytorch-cu130", marker = "sys_platform == 'win32' or sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

- [ ] **Step 4: Lock, prove no existing version moved, and regenerate the export**

```bash
cp uv.lock "$TMP/uv.lock.before"
/c/Users/ghimi/.local/bin/uv.exe lock
.venv/Scripts/python.exe - "$TMP/uv.lock.before" uv.lock <<'EOF'
import sys, tomllib
before, after = ({p["name"]: p["version"] for p in tomllib.load(open(path, "rb"))["package"]} for path in sys.argv[1:])
changed = {name: (before[name], after[name]) for name in before if name in after and before[name] != after[name]}
print("changed:", changed, "removed:", sorted(set(before) - set(after)), "added:", sorted(set(after) - set(before)))
EOF
/c/Users/ghimi/.local/bin/uv.exe export --frozen --no-default-groups --no-hashes --format requirements-txt --output-file requirements_labeler.txt
git diff requirements_labeler.txt
```

Expected: `changed: {}` and `removed: []`; the export diff touches only the generating-command comment line.

- [ ] **Step 5: Install into the existing environment**

```bash
/c/Users/ghimi/.local/bin/uv.exe sync --frozen --dry-run
```

If the dry run would uninstall `pip`, the labeler launcher would break, so use `--inexact` instead and record that in the workflow document in Task 10. Then:

```bash
/c/Users/ghimi/.local/bin/uv.exe sync --frozen
.venv/Scripts/python.exe -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv/Scripts/python.exe -m pip --version
```

Expected: `2.14.1+cu130 0.29.1+cu130 True NVIDIA GeForce RTX 4080 ...`, and pip still present.

- [ ] **Step 6: Run the test to see it pass, then the whole suite**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock requirements_labeler.txt tests/test_repository_reproducibility.py
git commit -m "build: add PyTorch as a default model dependency group"
```

---

### Task 2: Training configuration contract

**Files:**
- Modify: `src/frog_classifier/data/config.py` (rename `_section` to `require_section` and `_value` to `require_value`)
- Create: `config/training.toml`
- Create: `src/frog_classifier/training/__init__.py`
- Create: `src/frog_classifier/training/config.py`
- Create: `tests/training/__init__.py`
- Create: `tests/training/helpers.py`
- Test: `tests/training/test_config.py`

**Interfaces:**
- Consumes: `frog_classifier.data.config.ConfigError`.
- Produces: `load_training_config(path: Path) -> TrainingConfig`; dataclasses `TrainingConfig(seed: int, pretrained: bool, optimization: OptimizationConfig, augmentation: AugmentationConfig, baseline: BaselineConfig, source_path: Path, sha256: str)`, `OptimizationConfig(batch_size, max_epochs, learning_rate, weight_decay, early_stopping_patience)`, `AugmentationConfig(time_mask_max_frames, frequency_mask_max_bins, max_attenuation_db)`, `BaselineConfig(image_size, inverse_regularization)`; test helper `TRAINING_CONFIG: bytes`.

- [ ] **Step 1: Share the TOML validation helpers**

In `src/frog_classifier/data/config.py`, rename every `_section(` to `require_section(` and every `_value(` to `require_value(`, including the two `def` lines. They become the shared validators for both configuration contracts.

Run: `.venv/Scripts/python.exe -m unittest tests.data.test_config`
Expected: PASS.

- [ ] **Step 2: Write the test helper and failing tests**

`tests/training/__init__.py`: empty file.

`tests/training/helpers.py`:

```python
from __future__ import annotations


TRAINING_CONFIG = b'''schema_version = 1
seed = 7

[model]
pretrained = false

[optimization]
batch_size = 4
max_epochs = 2
learning_rate = 0.001
weight_decay = 0.0
early_stopping_patience = 1

[augmentation]
time_mask_max_frames = 4
frequency_mask_max_bins = 4
max_attenuation_db = 3.0

[baseline]
image_size = 8
inverse_regularization = 1.0
'''
```

`tests/training/test_config.py`:

```python
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from frog_classifier.data.config import ConfigError
from frog_classifier.training.config import (
    AugmentationConfig,
    BaselineConfig,
    OptimizationConfig,
    load_training_config,
)
from tests.training.helpers import TRAINING_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]


class TrainingConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "training.toml"

    def _load(self, content: bytes):
        self.path.write_bytes(content)
        return load_training_config(self.path)

    def test_loads_every_value_and_its_checksum(self) -> None:
        config = self._load(TRAINING_CONFIG)

        self.assertEqual(config.seed, 7)
        self.assertFalse(config.pretrained)
        self.assertEqual(config.optimization, OptimizationConfig(4, 2, 0.001, 0.0, 1))
        self.assertEqual(config.augmentation, AugmentationConfig(4, 4, 3.0))
        self.assertEqual(config.baseline, BaselineConfig(8, 1.0))
        self.assertEqual(config.sha256, hashlib.sha256(TRAINING_CONFIG).hexdigest())
        self.assertEqual(config.source_path, self.path.resolve())

    def test_tracked_configuration_matches_the_design(self) -> None:
        config = load_training_config(REPO_ROOT / "config" / "training.toml")

        self.assertEqual(config.seed, 1337)
        self.assertTrue(config.pretrained)
        self.assertEqual(config.optimization, OptimizationConfig(64, 30, 0.0003, 0.01, 8))
        self.assertEqual(config.augmentation, AugmentationConfig(20, 12, 6.0))
        self.assertEqual(config.baseline, BaselineConfig(64, 1.0))

    def test_rejects_another_schema_version(self) -> None:
        with self.assertRaisesRegex(ConfigError, "schema_version"):
            self._load(TRAINING_CONFIG.replace(b"schema_version = 1", b"schema_version = 2"))

    def test_rejects_unknown_and_missing_keys(self) -> None:
        cases = {
            "unknown top-level key": TRAINING_CONFIG + b"\nextra = 1\n",
            "unknown section key": TRAINING_CONFIG.replace(b"pretrained = false", b"pretrained = false\ndepth = 18"),
            "missing section": TRAINING_CONFIG.split(b"[baseline]")[0],
        }
        for name, content in cases.items():
            with self.subTest(name), self.assertRaises(ConfigError):
                self._load(content)

    def test_rejects_wrong_types(self) -> None:
        cases = {
            "fractional batch": (b"batch_size = 4", b"batch_size = 4.5"),
            "text flag": (b"pretrained = false", b'pretrained = "no"'),
            "Boolean seed": (b"seed = 7", b"seed = true"),
        }
        for name, (old, new) in cases.items():
            with self.subTest(name), self.assertRaises(ConfigError):
                self._load(TRAINING_CONFIG.replace(old, new))

    def test_rejects_out_of_range_and_non_finite_values(self) -> None:
        cases = (
            (b"seed = 7", b"seed = -1"),
            (b"batch_size = 4", b"batch_size = 0"),
            (b"max_epochs = 2", b"max_epochs = 0"),
            (b"early_stopping_patience = 1", b"early_stopping_patience = 0"),
            (b"learning_rate = 0.001", b"learning_rate = 0.0"),
            (b"learning_rate = 0.001", b"learning_rate = nan"),
            (b"weight_decay = 0.0", b"weight_decay = -0.1"),
            (b"time_mask_max_frames = 4", b"time_mask_max_frames = -1"),
            (b"frequency_mask_max_bins = 4", b"frequency_mask_max_bins = -1"),
            (b"max_attenuation_db = 3.0", b"max_attenuation_db = -1.0"),
            (b"image_size = 8", b"image_size = 0"),
            (b"inverse_regularization = 1.0", b"inverse_regularization = 0.0"),
        )
        for old, new in cases:
            with self.subTest(new.decode()), self.assertRaises(ConfigError):
                self._load(TRAINING_CONFIG.replace(old, new))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `.venv/Scripts/python.exe -m unittest tests.training.test_config -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'frog_classifier.training'`.

- [ ] **Step 4: Write the configuration and loader**

`config/training.toml`:

```toml
schema_version = 1
seed = 1337

[model]
pretrained = true

[optimization]
batch_size = 64
max_epochs = 30
learning_rate = 0.0003
weight_decay = 0.01
early_stopping_patience = 8

[augmentation]
time_mask_max_frames = 20
frequency_mask_max_bins = 12
max_attenuation_db = 6.0

[baseline]
image_size = 64
inverse_regularization = 1.0
```

`src/frog_classifier/training/__init__.py`:

```python
"""Model training, evaluation, saved runs, and their verification.

Nothing is re-exported here: most modules import PyTorch, and the baseline
module must stay importable without it. Import modules directly.
"""
```

`src/frog_classifier/training/config.py`:

```python
"""The training contract in config/training.toml, loaded and validated."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from frog_classifier.data.config import ConfigError, require_section, require_value


SUPPORTED_SCHEMA_VERSION = 1
_TOP_LEVEL_KEYS = {"schema_version", "seed", "model", "optimization", "augmentation", "baseline"}


@dataclass(frozen=True)
class OptimizationConfig:
    batch_size: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int


@dataclass(frozen=True)
class AugmentationConfig:
    time_mask_max_frames: int
    frequency_mask_max_bins: int
    max_attenuation_db: float


@dataclass(frozen=True)
class BaselineConfig:
    image_size: int
    inverse_regularization: float


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    pretrained: bool
    optimization: OptimizationConfig
    augmentation: AugmentationConfig
    baseline: BaselineConfig
    source_path: Path
    sha256: str


def load_training_config(path: Path) -> TrainingConfig:
    source = path.read_bytes()
    try:
        data = tomllib.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"invalid training TOML: {error}") from error

    if "schema_version" in data and require_value(data, "schema_version", int) != SUPPORTED_SCHEMA_VERSION:
        raise ConfigError(f"training schema_version must be {SUPPORTED_SCHEMA_VERSION}")
    if set(data) != _TOP_LEVEL_KEYS:
        raise ConfigError(
            f"invalid training keys: {sorted(data)}; "
            f"missing: {sorted(_TOP_LEVEL_KEYS - set(data))}; "
            f"unexpected: {sorted(set(data) - _TOP_LEVEL_KEYS)}"
        )

    model_data = require_section(data, "model", {"pretrained"})
    optimization_data = require_section(data, "optimization", {
        "batch_size", "max_epochs", "learning_rate", "weight_decay", "early_stopping_patience",
    })
    augmentation_data = require_section(data, "augmentation", {
        "time_mask_max_frames", "frequency_mask_max_bins", "max_attenuation_db",
    })
    baseline_data = require_section(data, "baseline", {"image_size", "inverse_regularization"})

    seed = int(require_value(data, "seed", int))
    pretrained = bool(require_value(model_data, "pretrained", bool))
    optimization = OptimizationConfig(
        batch_size=int(require_value(optimization_data, "batch_size", int)),
        max_epochs=int(require_value(optimization_data, "max_epochs", int)),
        learning_rate=float(require_value(optimization_data, "learning_rate", float)),
        weight_decay=float(require_value(optimization_data, "weight_decay", float)),
        early_stopping_patience=int(require_value(optimization_data, "early_stopping_patience", int)),
    )
    augmentation = AugmentationConfig(
        time_mask_max_frames=int(require_value(augmentation_data, "time_mask_max_frames", int)),
        frequency_mask_max_bins=int(require_value(augmentation_data, "frequency_mask_max_bins", int)),
        max_attenuation_db=float(require_value(augmentation_data, "max_attenuation_db", float)),
    )
    baseline = BaselineConfig(
        image_size=int(require_value(baseline_data, "image_size", int)),
        inverse_regularization=float(require_value(baseline_data, "inverse_regularization", float)),
    )

    if seed < 0:
        raise ConfigError("seed must be nonnegative")
    if min(optimization.batch_size, optimization.max_epochs, optimization.early_stopping_patience) < 1:
        raise ConfigError("batch_size, max_epochs, and early_stopping_patience must be at least 1")
    if optimization.learning_rate <= 0 or optimization.weight_decay < 0:
        raise ConfigError("learning_rate must be positive and weight_decay nonnegative")
    if (
        min(augmentation.time_mask_max_frames, augmentation.frequency_mask_max_bins) < 0
        or augmentation.max_attenuation_db < 0
    ):
        raise ConfigError("augmentation limits must be nonnegative")
    if baseline.image_size < 1 or baseline.inverse_regularization <= 0:
        raise ConfigError("baseline image_size must be at least 1 and inverse_regularization positive")

    return TrainingConfig(
        seed=seed,
        pretrained=pretrained,
        optimization=optimization,
        augmentation=augmentation,
        baseline=baseline,
        source_path=path.resolve(),
        sha256=hashlib.sha256(source).hexdigest(),
    )
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `.venv/Scripts/python.exe -m unittest tests.training.test_config tests.data.test_config -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add config/training.toml src/frog_classifier/data/config.py src/frog_classifier/training tests/training
git commit -m "feat: add the training configuration contract"
```

---

### Task 3: Image loading and the shared baseline

**Files:**
- Create: `src/frog_classifier/training/images.py`
- Create: `src/frog_classifier/training/baseline.py`
- Modify: `scripts/train_baseline.py`
- Modify: `tests/test_train_baseline.py` (patch the classifier class directly)
- Test: `tests/training/test_images.py`, `tests/training/test_baseline.py`

**Interfaces:**
- Consumes: `PreprocessingConfig` from `frog_classifier.data.config`.
- Produces: `expected_image_shape(config: PreprocessingConfig) -> tuple[int, int]`; `load_spectrograms(paths: Sequence[Path], shape: tuple[int, int]) -> LoadedImages` with `LoadedImages.pixels: np.ndarray (count, height, width) uint8` and `LoadedImages.sha256: tuple[str, ...]`; `baseline_features(images: Iterable[Image.Image], *, size: int) -> np.ndarray`; `fit_baseline(features: np.ndarray, labels: np.ndarray, *, inverse_regularization: float) -> LogisticRegression`.

- [ ] **Step 1: Write the failing tests**

`tests/training/test_images.py`:

```python
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from frog_classifier.training.images import expected_image_shape, load_spectrograms
from tests.data.helpers import load_test_config


class ImageLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _write(self, name: str, image: Image.Image) -> Path:
        path = self.root / name
        image.save(path)
        return path

    def test_contract_shape_is_mels_by_centred_frames(self) -> None:
        self.assertEqual(expected_image_shape(load_test_config(self.root)), (128, 216))

    def test_loads_images_in_order_with_their_checksums(self) -> None:
        first = self._write("a.png", Image.fromarray(np.full((4, 6), 10, dtype=np.uint8)))
        second = self._write("b.png", Image.fromarray(np.arange(24, dtype=np.uint8).reshape(4, 6)))

        loaded = load_spectrograms([second, first], (4, 6))

        self.assertEqual(loaded.pixels.dtype, np.uint8)
        np.testing.assert_array_equal(loaded.pixels[0], np.arange(24).reshape(4, 6))
        np.testing.assert_array_equal(loaded.pixels[1], np.full((4, 6), 10))
        self.assertEqual(loaded.sha256, (
            hashlib.sha256(second.read_bytes()).hexdigest(),
            hashlib.sha256(first.read_bytes()).hexdigest(),
        ))

    def test_rejects_a_wrong_size_naming_the_file(self) -> None:
        path = self._write("small.png", Image.fromarray(np.zeros((3, 6), dtype=np.uint8)))

        with self.assertRaisesRegex(ValueError, "small.png"):
            load_spectrograms([path], (4, 6))

    def test_rejects_a_colour_image(self) -> None:
        path = self._write("colour.png", Image.new("RGB", (6, 4)))

        with self.assertRaisesRegex(ValueError, "grayscale"):
            load_spectrograms([path], (4, 6))


if __name__ == "__main__":
    unittest.main()
```

`tests/training/test_baseline.py`:

```python
from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from frog_classifier.training.baseline import baseline_features, fit_baseline


class BaselineTests(unittest.TestCase):
    def test_features_are_resized_scaled_and_flattened(self) -> None:
        images = [
            Image.fromarray(np.full((4, 6), 255, dtype=np.uint8)),
            Image.fromarray(np.zeros((4, 6), dtype=np.uint8)),
        ]

        features = baseline_features(images, size=2)

        self.assertEqual(features.shape, (2, 4))
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_array_equal(features[0], np.ones(4))
        np.testing.assert_array_equal(features[1], np.zeros(4))

    def test_fits_a_balanced_logistic_regression(self) -> None:
        features = np.array([[0.0], [0.1], [0.2], [0.9], [1.0]])
        labels = np.array([0, 0, 0, 1, 1])

        classifier = fit_baseline(features, labels, inverse_regularization=10.0)

        self.assertEqual(classifier.C, 10.0)
        self.assertEqual(classifier.class_weight, "balanced")
        np.testing.assert_array_equal(classifier.predict(features), labels)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python.exe -m unittest tests.training.test_images tests.training.test_baseline -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the modules**

`src/frog_classifier/training/images.py`:

```python
"""Load stored spectrogram images exactly as they are on disk."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from frog_classifier.data.config import PreprocessingConfig


@dataclass(frozen=True)
class LoadedImages:
    pixels: np.ndarray
    sha256: tuple[str, ...]


def expected_image_shape(config: PreprocessingConfig) -> tuple[int, int]:
    """(height, width) of every stored spectrogram under the preprocessing contract.

    The width follows librosa's centred framing: one frame per hop, plus one.
    """
    samples_per_chunk = config.audio.sample_rate_hz * config.audio.chunk_seconds
    return config.spectrogram.n_mels, 1 + samples_per_chunk // config.spectrogram.hop_length


def load_spectrograms(paths: Sequence[Path], shape: tuple[int, int]) -> LoadedImages:
    """Decode 8-bit grayscale PNGs into one (count, height, width) array, in order,
    with the SHA-256 of each file's bytes so a run records exactly what it read."""
    pixels = np.empty((len(paths), *shape), dtype=np.uint8)
    checksums: list[str] = []
    for index, path in enumerate(paths):
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            if image.mode != "L":
                raise ValueError(f"{path}: expected an 8-bit grayscale image, found mode {image.mode}")
            array = np.asarray(image)
        if array.shape != shape:
            raise ValueError(
                f"{path}: expected {shape[0]} by {shape[1]} pixels (height by width), "
                f"found {array.shape[0]} by {array.shape[1]}"
            )
        pixels[index] = array
        checksums.append(hashlib.sha256(data).hexdigest())
    return LoadedImages(pixels=pixels, sha256=tuple(checksums))
```

`src/frog_classifier/training/baseline.py`:

```python
"""Logistic-regression baseline: a pipeline sanity check and the bar a deeper model must clear."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression


def baseline_features(images: Iterable[Image.Image], *, size: int) -> np.ndarray:
    """Each image as grayscale, resized to size by size, scaled to [0, 1], and flattened to one row."""
    rows = [
        np.asarray(image.convert("L").resize((size, size)), dtype=np.float32).reshape(-1) / 255.0
        for image in images
    ]
    return np.stack(rows, axis=0)


def fit_baseline(features: np.ndarray, labels: np.ndarray, *, inverse_regularization: float) -> LogisticRegression:
    classifier = LogisticRegression(C=inverse_regularization, max_iter=2000, class_weight="balanced")
    return classifier.fit(features, labels)
```

- [ ] **Step 4: Run them to see them pass**

Run: `.venv/Scripts/python.exe -m unittest tests.training.test_images tests.training.test_baseline -v`
Expected: PASS.

- [ ] **Step 5: Point the baseline script at the shared code**

In `scripts/train_baseline.py`:
- Replace `from sklearn.linear_model import LogisticRegression` with `from frog_classifier.training.baseline import baseline_features, fit_baseline`.
- Replace the body of `featurize` and add a generator after it:

```python
def featurize(paths: Iterable[Path], *, size: int) -> np.ndarray:
    return baseline_features(_grayscale_images(paths), size=size)


def _grayscale_images(paths: Iterable[Path]) -> Iterable[Image.Image]:
    for path in paths:
        with Image.open(path) as image:
            yield image.convert("L")
```

- Replace the classifier construction and fit (the `clf = LogisticRegression(...)` block and `clf.fit(X_train, y_train)`) with:

```python
    clf = fit_baseline(X_train, y_train, inverse_regularization=float(args.C))
```

In `tests/test_train_baseline.py`, add `from sklearn.linear_model import LogisticRegression` to the imports and replace both `patch.object(module.LogisticRegression, "fit", ...)` calls with `patch.object(LogisticRegression, "fit", ...)`, keeping their other arguments. The patch replaces the method on the class itself, which is the object the package now constructs.

- [ ] **Step 6: Run the baseline script tests**

Run: `.venv/Scripts/python.exe -m unittest tests.test_train_baseline -v`
Expected: PASS, six tests.

- [ ] **Step 7: Commit**

```bash
git add src/frog_classifier/training scripts/train_baseline.py tests/test_train_baseline.py tests/training
git commit -m "feat: load stored spectrograms and share the baseline featurization"
```

---

### Task 4: Training-batch augmentation

**Files:**
- Create: `src/frog_classifier/training/augment.py`
- Test: `tests/training/test_augment.py`

**Interfaces:**
- Consumes: `AugmentationConfig` from Task 2.
- Produces: `augment_batch(images: torch.Tensor, config: AugmentationConfig, *, db_floor: float, db_ceiling: float, generator: torch.Generator) -> torch.Tensor`, where images are float (batch, 1, mels, frames) in [0, 1] on the CPU.

- [ ] **Step 1: Write the failing tests**

`tests/training/test_augment.py`:

```python
from __future__ import annotations

import unittest

import torch

from frog_classifier.training.augment import augment_batch
from frog_classifier.training.config import AugmentationConfig


def _generator(seed: int = 1) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _augment(images: torch.Tensor, config: AugmentationConfig, seed: int = 1) -> torch.Tensor:
    return augment_batch(images, config, db_floor=0.0, db_ceiling=30.0, generator=_generator(seed))


class AugmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.images = torch.rand((6, 1, 16, 24), generator=_generator(0))

    def test_preserves_shape_type_and_range_without_touching_the_input(self) -> None:
        original = self.images.clone()

        augmented = _augment(self.images, AugmentationConfig(5, 4, 6.0))

        self.assertEqual(augmented.shape, self.images.shape)
        self.assertEqual(augmented.dtype, torch.float32)
        self.assertGreaterEqual(float(augmented.min()), 0.0)
        self.assertLessEqual(float(augmented.max()), 1.0)
        self.assertTrue(torch.equal(self.images, original))

    def test_the_same_seed_gives_the_same_batch(self) -> None:
        config = AugmentationConfig(5, 4, 6.0)

        self.assertTrue(torch.equal(_augment(self.images, config, 3), _augment(self.images, config, 3)))
        self.assertFalse(torch.equal(_augment(self.images, config, 3), _augment(self.images, config, 4)))

    def test_time_shift_rolls_each_image_along_time(self) -> None:
        augmented = _augment(self.images, AugmentationConfig(0, 0, 0.0))

        for original, shifted in zip(self.images, augmented):
            with self.subTest():
                self.assertTrue(any(
                    torch.equal(shifted, torch.roll(original, offset, dims=2))
                    for offset in range(original.shape[2])
                ))

    def test_attenuation_never_brightens_and_keeps_the_noise_floor(self) -> None:
        # Constant along time so the shift cannot move anything.
        rows = torch.rand((6, 1, 16, 1), generator=_generator(5))
        rows[:, :, :4] = 0.0
        rows[:, :, 4] = 1.0
        images = rows.expand(-1, -1, -1, 24).contiguous()

        augmented = _augment(images, AugmentationConfig(0, 0, 6.0))

        self.assertTrue(bool((augmented <= images + 1e-6).all()))
        self.assertTrue(bool((augmented[images == 0.0] == 0.0).all()))
        self.assertTrue(bool((augmented < images - 1e-3).any()))
        # A pixel at the 30 dB ceiling drops by at most the 6 dB limit.
        self.assertTrue(bool((augmented[:, :, 4] >= 24.0 / 30.0 - 1e-5).all()))

    def test_masks_set_whole_time_columns_and_mel_rows_to_the_floor(self) -> None:
        images = torch.ones((8, 1, 16, 24))

        augmented = _augment(images, AugmentationConfig(5, 3, 0.0))

        for image in augmented[:, 0]:
            zero_columns = (image == 0).all(dim=0)
            zero_rows = (image == 0).all(dim=1)
            with self.subTest():
                self.assertTrue(torch.equal(image == 0, zero_columns[None, :] | zero_rows[:, None]))
                self.assertLessEqual(int(zero_columns.sum()), 5)
                self.assertLessEqual(int(zero_rows.sum()), 3)
                for indices in (zero_columns.nonzero().flatten(), zero_rows.nonzero().flatten()):
                    if len(indices):
                        self.assertEqual(int(indices[-1] - indices[0]) + 1, len(indices))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python.exe -m unittest tests.training.test_augment -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the module**

`src/frog_classifier/training/augment.py`:

```python
"""Seeded augmentation of training batches.

Images are float tensors shaped (batch, 1, mels, frames) with values in [0, 1],
where 0 is the display floor (the recording's noise floor under the current
contract) and 1 is the display ceiling.
"""

from __future__ import annotations

import torch

from .config import AugmentationConfig


def augment_batch(
    images: torch.Tensor,
    config: AugmentationConfig,
    *,
    db_floor: float,
    db_ceiling: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return an augmented copy. Draws come from the generator in a fixed order."""
    augmented = _time_shift(images, generator)
    augmented = _attenuate(augmented, config.max_attenuation_db, db_floor, db_ceiling, generator)
    augmented = _mask(augmented, config.time_mask_max_frames, dim=3, generator=generator)
    return _mask(augmented, config.frequency_mask_max_bins, dim=2, generator=generator)


def _time_shift(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Roll every image circularly along time, because a call can start anywhere in a window."""
    batch, _, _, frames = images.shape
    offsets = torch.randint(0, frames, (batch,), generator=generator)
    columns = (torch.arange(frames).unsqueeze(0) - offsets.unsqueeze(1)) % frames
    return torch.gather(images, 3, columns[:, None, None, :].expand_as(images))


def _attenuate(
    images: torch.Tensor,
    max_db: float,
    db_floor: float,
    db_ceiling: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Make each image quieter by up to max_db, as if its sources were farther away.

    Pixels store decibels above the noise floor. The energy above the floor is
    scaled down in the power domain, so the floor itself stays where it is.
    """
    if max_db == 0:
        return images
    db_range = db_ceiling - db_floor
    attenuation_db = torch.rand(images.shape[0], generator=generator) * max_db
    gain = torch.pow(10.0, -attenuation_db / 10.0).view(-1, 1, 1, 1)
    level_db = db_floor + images * db_range
    excess = torch.pow(10.0, level_db / 10.0) - 1.0
    quieter_db = 10.0 * torch.log10(1.0 + excess * gain)
    return ((quieter_db - db_floor) / db_range).clamp(0.0, 1.0)


def _mask(images: torch.Tensor, max_width: int, *, dim: int, generator: torch.Generator) -> torch.Tensor:
    """Set one random block of up to max_width positions along dim to the floor."""
    if max_width == 0:
        return images
    batch, size = images.shape[0], images.shape[dim]
    widths = torch.randint(0, min(max_width, size) + 1, (batch,), generator=generator)
    starts = (torch.rand(batch, generator=generator) * (size - widths + 1)).floor().long()
    positions = torch.arange(size)
    masked = (positions >= starts[:, None]) & (positions < (starts + widths)[:, None])
    shape = [batch, 1, 1, 1]
    shape[dim] = size
    return images.masked_fill(masked.view(shape), 0.0)
```

- [ ] **Step 4: Run them to see them pass**

Run: `.venv/Scripts/python.exe -m unittest tests.training.test_augment -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/training/augment.py tests/training/test_augment.py
git commit -m "feat: add seeded spectrogram augmentation for training batches"
```

---

### Task 5: Model and training loop

**Files:**
- Create: `src/frog_classifier/training/model.py`
- Create: `src/frog_classifier/training/fit.py`
- Test: `tests/training/test_model.py`, `tests/training/test_fit.py`

**Interfaces:**
- Consumes: `TrainingConfig` (Task 2), `augment_batch` (Task 4).
- Produces: `ARCHITECTURE = "resnet18"`, `PRETRAINED_WEIGHTS = "IMAGENET1K_V1"`, `SpectrogramResNet(*, pretrained: bool)`, `to_model_input(pixels) -> torch.Tensor`; `select_device(requested: str | None = None) -> torch.device`, `use_full_precision() -> None`, `fit_model(train_pixels, train_labels, val_pixels, val_labels, config, *, db_floor, db_ceiling, device, progress=print) -> FitResult(model, best_epoch, history)`, `EpochRecord(epoch, train_loss, val_loss, val_average_precision, learning_rate, seconds)`, `predict_logits(model, pixels, device) -> np.ndarray`, `predict_probabilities(model, pixels, device) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

`tests/training/test_model.py`:

```python
from __future__ import annotations

import unittest

import numpy as np
import torch

from frog_classifier.training.model import SpectrogramResNet, to_model_input


class ModelTests(unittest.TestCase):
    def test_returns_one_logit_per_image(self) -> None:
        model = SpectrogramResNet(pretrained=False).eval()

        with torch.inference_mode():
            logits = model(torch.rand((3, 1, 32, 48)))

        self.assertEqual(logits.shape, (3,))

    def test_rejects_images_without_one_channel(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch, 1"):
            SpectrogramResNet(pretrained=False)(torch.rand((2, 3, 32, 48)))

    def test_a_reloaded_state_dictionary_reproduces_outputs(self) -> None:
        torch.manual_seed(0)
        original = SpectrogramResNet(pretrained=False).eval()
        copy = SpectrogramResNet(pretrained=False)
        copy.load_state_dict(original.state_dict())
        copy.eval()
        images = torch.rand((2, 1, 32, 48))

        with torch.inference_mode():
            self.assertTrue(torch.equal(original(images), copy(images)))

    def test_model_input_scales_eight_bit_pixels(self) -> None:
        pixels = np.array([[[0, 255], [51, 102]]], dtype=np.uint8)

        tensor = to_model_input(pixels)

        self.assertEqual(tensor.shape, (1, 1, 2, 2))
        self.assertEqual(tensor.dtype, torch.float32)
        torch.testing.assert_close(tensor[0, 0], torch.tensor([[0.0, 1.0], [0.2, 0.4]]))

    def test_model_input_rejects_other_types(self) -> None:
        with self.assertRaises(ValueError):
            to_model_input(np.zeros((1, 2, 2), dtype=np.float32))
```

`tests/training/test_fit.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from frog_classifier.training.config import load_training_config
from frog_classifier.training.fit import fit_model, predict_probabilities, select_device
from tests.training.helpers import TRAINING_CONFIG


def _images(count: int, frog: bool, seed: int) -> np.ndarray:
    pixels = np.random.default_rng(seed).integers(0, 40, (count, 32, 48), dtype=np.uint8)
    if frog:
        pixels[:, 12:16, :] = 220
    return pixels


class FitTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "training.toml"
        path.write_bytes(TRAINING_CONFIG)
        self.config = load_training_config(path)

    def _fit(self, train_count: int, messages: list[str]):
        half = train_count // 2
        train = np.concatenate([_images(half, True, 1), _images(train_count - half, False, 2)])
        labels = np.array([1] * half + [0] * (train_count - half))
        val = np.concatenate([_images(2, True, 3), _images(2, False, 4)])
        return fit_model(
            train, labels, val, np.array([1, 1, 0, 0]), self.config,
            db_floor=0.0, db_ceiling=30.0, device=torch.device("cpu"), progress=messages.append,
        )

    def test_trains_keeps_the_best_epoch_and_reports_progress(self) -> None:
        messages: list[str] = []

        result = self._fit(8, messages)

        self.assertIn(len(result.history), (1, 2))
        self.assertIn(result.best_epoch, [record.epoch for record in result.history])
        self.assertFalse(result.model.training)
        self.assertTrue(any(message.startswith("epoch") for message in messages))

    def test_trains_when_the_last_batch_would_hold_one_image(self) -> None:
        # Nine images in batches of four would leave a batch of one, which BatchNorm rejects.
        result = self._fit(9, [])

        self.assertGreaterEqual(len(result.history), 1)

    def test_rejects_fewer_than_two_training_images(self) -> None:
        with self.assertRaisesRegex(ValueError, "two images"):
            self._fit(1, [])

    def test_probabilities_are_in_input_order_and_batch_independent(self) -> None:
        result = self._fit(8, [])
        pixels = np.concatenate([_images(3, True, 9), _images(3, False, 10)])

        together = predict_probabilities(result.model, pixels, torch.device("cpu"))
        alone = np.array([
            predict_probabilities(result.model, pixels[index : index + 1], torch.device("cpu"))[0]
            for index in range(len(pixels))
        ])

        self.assertEqual(together.shape, (6,))
        self.assertTrue(((together >= 0) & (together <= 1)).all())
        np.testing.assert_allclose(together, alone, atol=1e-6)


class DeviceTests(unittest.TestCase):
    def test_falls_back_to_the_cpu_without_cuda(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(select_device(), torch.device("cpu"))

    def test_an_explicit_cuda_request_without_a_gpu_fails_clearly(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                select_device("cuda")

    def test_an_explicit_cpu_request_is_honoured(self) -> None:
        self.assertEqual(select_device("cpu"), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python.exe -m unittest tests.training.test_model tests.training.test_fit -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the model**

`src/frog_classifier/training/model.py`:

```python
"""ResNet-18 for one-channel spectrogram images."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


ARCHITECTURE = "resnet18"
PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class SpectrogramResNet(nn.Module):
    """Takes images shaped (batch, 1, mels, frames) in [0, 1]; returns one frog logit per image."""

    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
        # The pretrained filters expect three ImageNet-normalized colour channels.
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(f"expected images shaped (batch, 1, mels, frames), got {tuple(images.shape)}")
        colour = images.expand(-1, 3, -1, -1)
        return self.backbone((colour - self.mean) / self.std).squeeze(1)


def to_model_input(pixels: np.ndarray | torch.Tensor) -> torch.Tensor:
    """8-bit pixels shaped (batch, mels, frames) as float (batch, 1, mels, frames) in [0, 1]."""
    tensor = torch.as_tensor(pixels)
    if tensor.dtype != torch.uint8 or tensor.ndim != 3:
        raise ValueError("expected uint8 pixels shaped (batch, mels, frames)")
    return tensor.unsqueeze(1).float().div_(255.0)
```

- [ ] **Step 4: Write the training loop**

`src/frog_classifier/training/fit.py`:

```python
"""Training loop with early stopping, and batch prediction."""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from .augment import augment_batch
from .config import TrainingConfig
from .model import SpectrogramResNet, to_model_input


PREDICTION_BATCH_SIZE = 256


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_average_precision: float
    learning_rate: float
    seconds: float


@dataclass(frozen=True)
class FitResult:
    model: SpectrogramResNet
    best_epoch: int
    history: tuple[EpochRecord, ...]


def select_device(requested: str | None = None) -> torch.device:
    """The requested device, or the GPU when one is available and the CPU otherwise."""
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but no CUDA device is available")
    return device


def use_full_precision() -> None:
    """Keep GPU arithmetic in full fp32 so GPU and CPU predictions agree.

    TF32 is faster but keeps only about three significant digits, which would
    make a saved model's predictions depend on the hardware that runs it.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def fit_model(
    train_pixels: np.ndarray,
    train_labels: np.ndarray,
    val_pixels: np.ndarray,
    val_labels: np.ndarray,
    config: TrainingConfig,
    *,
    db_floor: float,
    db_ceiling: float,
    device: torch.device,
    progress: Callable[[str], None] = print,
) -> FitResult:
    """Train from the configured starting weights and return the best validation epoch's model."""
    if len(train_pixels) < 2:
        raise ValueError("training needs at least two images")
    _seed_everything(config.seed)
    use_full_precision()
    settings = config.optimization
    model = SpectrogramResNet(pretrained=config.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.max_epochs)
    loss_function = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(config.seed)
    pixels = torch.as_tensor(train_pixels)
    targets = torch.as_tensor(train_labels, dtype=torch.float32)
    val_targets = torch.as_tensor(val_labels, dtype=torch.float32)

    history: list[EpochRecord] = []
    best_state = copy.deepcopy(model.state_dict())
    best_precision = -math.inf
    best_epoch = 0
    for epoch in range(1, settings.max_epochs + 1):
        started = time.perf_counter()
        learning_rate = scheduler.get_last_lr()[0]
        model.train()
        seen, total_loss = 0, 0.0
        for index in _batches(len(pixels), settings.batch_size, generator):
            batch = augment_batch(
                to_model_input(pixels[index]),
                config.augmentation,
                db_floor=db_floor,
                db_ceiling=db_ceiling,
                generator=generator,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch.to(device)), targets[index].to(device))
            loss.backward()
            optimizer.step()
            seen += len(index)
            total_loss += loss.item() * len(index)
        scheduler.step()

        val_logits = predict_logits(model, val_pixels, device)
        val_loss = float(loss_function(torch.from_numpy(val_logits), val_targets))
        val_precision = float(average_precision_score(val_labels, val_logits))
        record = EpochRecord(epoch, total_loss / seen, val_loss, val_precision, learning_rate, time.perf_counter() - started)
        history.append(record)
        improved = val_precision > best_precision
        if improved:
            best_precision, best_epoch = val_precision, epoch
            best_state = copy.deepcopy(model.state_dict())
        progress(
            f"epoch {epoch:>2}  train loss {record.train_loss:.4f}  val loss {val_loss:.4f}  "
            f"val AP {val_precision:.4f}  {record.seconds:.1f}s{'  best' if improved else ''}"
        )
        if epoch - best_epoch >= settings.early_stopping_patience:
            progress(f"stopping: no validation improvement for {settings.early_stopping_patience} epochs")
            break

    model.load_state_dict(best_state)
    model.eval()
    return FitResult(model=model, best_epoch=best_epoch, history=tuple(history))


def predict_logits(model: nn.Module, pixels: np.ndarray, device: torch.device) -> np.ndarray:
    """One logit per image, without augmentation, in input order."""
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(pixels), PREDICTION_BATCH_SIZE):
            batch = to_model_input(pixels[start : start + PREDICTION_BATCH_SIZE]).to(device)
            outputs.append(model(batch).float().cpu())
    return torch.cat(outputs).numpy()


def predict_probabilities(model: nn.Module, pixels: np.ndarray, device: torch.device) -> np.ndarray:
    """Frog probability per image, in input order."""
    logits = torch.from_numpy(predict_logits(model, pixels, device)).double()
    return torch.sigmoid(logits).numpy()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _batches(count: int, batch_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    """Shuffled index batches. A final partial batch is dropped when other batches
    exist, because BatchNorm statistics from a handful of images are noisy and a
    batch of one fails outright."""
    batches = list(torch.split(torch.randperm(count, generator=generator), batch_size))
    if len(batches) > 1 and len(batches[-1]) < batch_size:
        batches.pop()
    return batches
```

If PyTorch 2.14 warns that `allow_tf32` is deprecated, set the replacement flags instead (`torch.backends.cudnn.conv.fp32_precision = "ieee"` and `torch.backends.cuda.matmul.fp32_precision = "ieee"`) and keep the docstring.

- [ ] **Step 5: Run them to see them pass**

Run: `.venv/Scripts/python.exe -W error::UserWarning -m unittest tests.training.test_model tests.training.test_fit -v`
Expected: PASS without warnings.

- [ ] **Step 6: Commit**

```bash
git add src/frog_classifier/training/model.py src/frog_classifier/training/fit.py tests/training/test_model.py tests/training/test_fit.py
git commit -m "feat: add the spectrogram ResNet-18 and its training loop"
```

---

### Task 6: Threshold rule and metrics

**Files:**
- Create: `src/frog_classifier/training/metrics.py`
- Test: `tests/training/test_metrics.py`

**Interfaces:**
- Consumes: `frog_classifier.labeling.queue.hour_of_day`.
- Produces: `THRESHOLD_RULE: str`; `select_threshold(labels, probabilities) -> float`; `evaluate(labels, probabilities, threshold) -> SplitMetrics`; `counts_by_hour(recording_ids, labels, probabilities, threshold) -> tuple[HourCounts, ...]`; `recall_by_faint(labels, faint, probabilities, threshold) -> dict[str, RecallCount]` with keys `"faint"` and `"other"`; `false_detections_per_hour(false_positive_rate: float, chunk_seconds: int) -> float`. Dataclass fields: `SplitMetrics(examples, frog, background, true_positives, false_positives, true_negatives, false_negatives, precision, recall, f1, false_positive_rate, average_precision, roc_auc)`, `HourCounts(hour, background, false_positives, frog, missed)`, `RecallCount(clips, detected)`.

- [ ] **Step 1: Write the failing tests**

`tests/training/test_metrics.py`:

```python
from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from frog_classifier.training.metrics import (
    HourCounts,
    RecallCount,
    counts_by_hour,
    evaluate,
    false_detections_per_hour,
    recall_by_faint,
    select_threshold,
)


class ThresholdTests(unittest.TestCase):
    def test_picks_the_threshold_with_the_highest_f1(self) -> None:
        # Thresholds 0.1, 0.35, 0.4, 0.8 give F1 0.67, 0.80, 0.50, 0.67.
        self.assertEqual(select_threshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.35)

    def test_ties_go_to_the_higher_threshold(self) -> None:
        # Thresholds 0.1 and 0.9 both give F1 2/3; nothing in between does better.
        self.assertEqual(select_threshold([1, 0, 0, 1], [0.1, 0.2, 0.3, 0.9]), 0.9)

    def test_needs_both_classes(self) -> None:
        with self.assertRaises(ValueError):
            select_threshold([1, 1], [0.2, 0.8])


class EvaluationTests(unittest.TestCase):
    def test_counts_and_rates_at_the_threshold(self) -> None:
        labels = [1, 1, 0, 0, 0]
        probabilities = [0.9, 0.2, 0.8, 0.1, 0.3]

        metrics = evaluate(labels, probabilities, 0.5)

        self.assertEqual(
            (metrics.examples, metrics.frog, metrics.background),
            (5, 2, 3),
        )
        self.assertEqual(
            (metrics.true_positives, metrics.false_positives, metrics.true_negatives, metrics.false_negatives),
            (1, 1, 2, 1),
        )
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)
        self.assertAlmostEqual(metrics.f1, 0.5)
        self.assertAlmostEqual(metrics.false_positive_rate, 1 / 3)
        self.assertAlmostEqual(metrics.average_precision, average_precision_score(labels, probabilities))
        self.assertAlmostEqual(metrics.roc_auc, roc_auc_score(labels, probabilities))

    def test_a_probability_equal_to_the_threshold_counts_as_frog(self) -> None:
        self.assertEqual(evaluate([1, 0], [0.5, 0.1], 0.5).true_positives, 1)

    def test_no_detections_give_zero_precision_instead_of_failing(self) -> None:
        metrics = evaluate([1, 0], [0.2, 0.1], 0.5)

        self.assertEqual((metrics.precision, metrics.recall, metrics.f1), (0.0, 0.0, 0.0))

    def test_needs_both_classes(self) -> None:
        with self.assertRaises(ValueError):
            evaluate([0, 0], [0.2, 0.8], 0.5)


class BreakdownTests(unittest.TestCase):
    def test_counts_errors_by_recording_hour_with_unknown_hours_last(self) -> None:
        rows = counts_by_hour(
            ["20220101_230000", "20220101_010000", "20220101_010000", "no_timestamp"],
            [0, 1, 0, 1],
            [0.1, 0.2, 0.9, 0.8],
            0.5,
        )

        self.assertEqual(rows, (
            HourCounts(hour=1, background=1, false_positives=1, frog=1, missed=1),
            HourCounts(hour=23, background=1, false_positives=0, frog=0, missed=0),
            HourCounts(hour=None, background=0, false_positives=0, frog=1, missed=0),
        ))

    def test_recall_is_split_by_the_faint_tag(self) -> None:
        result = recall_by_faint([1, 1, 1, 0], [True, False, False, False], [0.9, 0.2, 0.8, 0.9], 0.5)

        self.assertEqual(result, {"faint": RecallCount(1, 1), "other": RecallCount(2, 1)})

    def test_false_detections_per_hour_use_the_window_length(self) -> None:
        self.assertAlmostEqual(false_detections_per_hour(0.01, 5), 7.2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python.exe -m unittest tests.training.test_metrics -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the module**

`src/frog_classifier/training/metrics.py`:

```python
"""Threshold rule and evaluation metrics for frog detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from frog_classifier.labeling.queue import hour_of_day


THRESHOLD_RULE = "maximum F1 on the validation fold, ties to the higher threshold"


@dataclass(frozen=True)
class SplitMetrics:
    examples: int
    frog: int
    background: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    average_precision: float
    roc_auc: float


@dataclass(frozen=True)
class HourCounts:
    hour: int | None
    background: int
    false_positives: int
    frog: int
    missed: int


@dataclass(frozen=True)
class RecallCount:
    clips: int
    detected: int


def select_threshold(labels: ArrayLike, probabilities: ArrayLike) -> float:
    """The probability that maximizes F1, predicting frog at or above it."""
    labels, probabilities = _checked(labels, probabilities)
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    precision, recall = precision[:-1], recall[:-1]
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)
    # Thresholds ascend, so the last maximum is the highest threshold among ties.
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def evaluate(labels: ArrayLike, probabilities: ArrayLike, threshold: float) -> SplitMetrics:
    labels, probabilities = _checked(labels, probabilities)
    detected = probabilities >= threshold
    frog = labels == 1
    true_positives = int(np.sum(detected & frog))
    false_positives = int(np.sum(detected & ~frog))
    true_negatives = int(np.sum(~detected & ~frog))
    false_negatives = int(np.sum(~detected & frog))
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    return SplitMetrics(
        examples=len(labels),
        frog=int(frog.sum()),
        background=int((~frog).sum()),
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=_ratio(2 * precision * recall, precision + recall),
        false_positive_rate=_ratio(false_positives, false_positives + true_negatives),
        average_precision=float(average_precision_score(labels, probabilities)),
        roc_auc=float(roc_auc_score(labels, probabilities)),
    )


def counts_by_hour(
    recording_ids: Sequence[str],
    labels: ArrayLike,
    probabilities: ArrayLike,
    threshold: float,
) -> tuple[HourCounts, ...]:
    """Errors per recording hour, in hour order, with recordings lacking a timestamp last."""
    detected = np.asarray(probabilities, dtype=float) >= threshold
    counts: dict[int | None, list[int]] = {}
    for recording_id, label, hit in zip(recording_ids, np.asarray(labels), detected, strict=True):
        row = counts.setdefault(hour_of_day(recording_id), [0, 0, 0, 0])
        if label == 1:
            row[2] += 1
            row[3] += int(not hit)
        else:
            row[0] += 1
            row[1] += int(hit)
    order = sorted(counts, key=lambda hour: (hour is None, hour or 0))
    return tuple(HourCounts(hour, *counts[hour]) for hour in order)


def recall_by_faint(
    labels: ArrayLike,
    faint: ArrayLike,
    probabilities: ArrayLike,
    threshold: float,
) -> dict[str, RecallCount]:
    """Detected frog clips among those tagged faint and among the others."""
    frog = np.asarray(labels) == 1
    tagged = np.asarray(faint, dtype=bool)
    detected = np.asarray(probabilities, dtype=float) >= threshold
    result = {}
    for name, mask in (("faint", frog & tagged), ("other", frog & ~tagged)):
        result[name] = RecallCount(clips=int(mask.sum()), detected=int((mask & detected).sum()))
    return result


def false_detections_per_hour(false_positive_rate: float, chunk_seconds: int) -> float:
    """Expected false detections in one hour of recording that holds no frog call."""
    return false_positive_rate * 3600 / chunk_seconds


def _checked(labels: ArrayLike, probabilities: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or labels.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional and the same length")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("evaluation needs both frog and background examples")
    return labels, probabilities


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
```

- [ ] **Step 4: Run them to see them pass**

Run: `.venv/Scripts/python.exe -m unittest tests.training.test_metrics -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/frog_classifier/training/metrics.py tests/training/test_metrics.py
git commit -m "feat: add the validation threshold rule and detection metrics"
```

---

### Task 7: Run folder and report

**Files:**
- Create: `src/frog_classifier/training/artifact.py`
- Create: `src/frog_classifier/training/report.py`
- Modify: `tests/training/helpers.py` (add `example_metadata()`)
- Test: `tests/training/test_artifact.py`, `tests/training/test_report.py`

**Interfaces:**
- Consumes: `evaluate` (Task 6).
- Produces: constants `ARTIFACT_SCHEMA_VERSION = 1`, `RUN_METADATA = "run.json"`, `MODEL_FILE = "model.pt"`, `PREDICTIONS_FILE = "predictions.csv"`, `PREDICTION_COLUMNS`; `file_sha256(path) -> str`; `start_run(models_root: Path, name: str) -> Path` (staging folder); `finish_run(staging: Path, metadata: Mapping[str, Any]) -> Path` (final folder); `read_run(run_dir) -> dict`; `check_files(run_dir, metadata) -> list[str]`; `write_csv(path, columns, rows)`; `read_csv(path) -> list[dict[str, str]]`; `CurveInput(title, labels, resnet, baseline, resnet_threshold, baseline_threshold)`; `write_precision_recall_figure(path, panels)`; `render_report(metadata) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/training/helpers.py`:

```python
def _split(frog: int, background: int, true_positives: int, false_positives: int) -> dict:
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / frog
    return {
        "examples": frog + background, "frog": frog, "background": background,
        "true_positives": true_positives, "false_positives": false_positives,
        "true_negatives": background - false_positives, "false_negatives": frog - true_positives,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
        "false_positive_rate": false_positives / background,
        "average_precision": 0.9, "roc_auc": 0.95,
    }


def example_metadata(*, cross_validation: bool) -> dict:
    """The run.json shape that render_report reads, with small hand-made numbers."""
    rotation = {
        "rotation": 0, "test_fold": 0, "validation_fold": 1, "best_epoch": 3,
        "resnet_threshold": 0.6, "baseline_threshold": 0.4,
        "resnet": _split(20, 50, 18, 2), "baseline": _split(20, 50, 12, 9),
    }
    spread = {"mean": 0.9, "std": 0.02, "min": 0.88, "max": 0.92}
    return {
        "artifact_schema_version": 1,
        "run_name": "resnet18-20261014T000000Z",
        "created_at": "2026-10-14T00:00:00Z",
        "model": {"architecture": "resnet18", "pretrained_weights": "IMAGENET1K_V1", "input_shape": [1, 128, 216]},
        "threshold": {"value": 0.6, "rule": "maximum F1 on the validation fold, ties to the higher threshold"},
        "seed": 1337,
        "training": {"best_epoch": 3, "epochs_run": 7, "device": "cpu", "device_name": "CPU", "seconds": 42.0},
        "data": {
            "examples": 210, "test_fold": 0, "validation_fold": 1, "faint_tagged_frog_clips": 4,
            "splits": {
                "train": {"examples": 70, "frog": 20, "background": 50, "recordings": 9},
                "validation": {"examples": 70, "frog": 20, "background": 50, "recordings": 8},
                "test": {"examples": 70, "frog": 20, "background": 50, "recordings": 7},
            },
        },
        "checksums": {"manifest": "a" * 64, "preprocessing_config": "b" * 64, "training_config": "c" * 64, "uv_lock": "d" * 64},
        "environment": {"python": "3.13.3", "torch": "2.14.1", "torchvision": "0.29.1", "numpy": "2.3.5", "cuda": None, "cudnn": None, "platform": "test"},
        "source": {"git_commit": "f" * 40, "git_worktree_dirty": False},
        "metrics": {
            "resnet": {"validation": _split(20, 50, 17, 3), "test": _split(20, 50, 18, 2)},
            "baseline": {"threshold": 0.4, "validation": _split(20, 50, 11, 8), "test": _split(20, 50, 12, 9)},
            "false_detections_per_hour": 28.8,
            "test_by_hour": [
                {"hour": 1, "background": 30, "false_positives": 2, "frog": 15, "missed": 1},
                {"hour": None, "background": 20, "false_positives": 0, "frog": 5, "missed": 1},
            ],
            "faint_recall": {
                "validation": {"faint": {"clips": 2, "detected": 1}, "other": {"clips": 18, "detected": 16}},
                "test": {"faint": {"clips": 2, "detected": 2}, "other": {"clips": 18, "detected": 16}},
            },
        },
        "cross_validation": (
            {
                "rotations": [rotation],
                "summary": {
                    model: {metric: spread for metric in ("average_precision", "precision", "recall", "f1")}
                    for model in ("resnet", "baseline")
                },
            }
            if cross_validation
            else None
        ),
    }
```

`tests/training/test_artifact.py`:

```python
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from frog_classifier.training.artifact import (
    RUN_METADATA,
    check_files,
    finish_run,
    read_csv,
    read_run,
    start_run,
    write_csv,
)


class RunFolderTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.models = Path(directory.name) / "models"

    def test_a_run_is_staged_under_a_hidden_name_then_renamed_with_checksums(self) -> None:
        staging = start_run(self.models, "resnet18-x")
        (staging / "a.txt").write_bytes(b"alpha")

        final = finish_run(staging, {"run_name": "resnet18-x"})

        self.assertEqual(final, self.models / "resnet18-x")
        self.assertFalse(staging.exists())
        metadata = read_run(final)
        self.assertEqual(metadata["run_name"], "resnet18-x")
        self.assertEqual(metadata["files"], {"a.txt": hashlib.sha256(b"alpha").hexdigest()})
        self.assertEqual(check_files(final, metadata), [])
        self.assertNotIn(b"\r", (final / RUN_METADATA).read_bytes())
        self.assertEqual(json.loads((final / RUN_METADATA).read_text(encoding="utf-8")), metadata)

    def test_refuses_a_name_that_is_already_used_or_staged(self) -> None:
        (self.models / "resnet18-x").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            start_run(self.models, "resnet18-x")

        start_run(self.models, "resnet18-y")
        with self.assertRaises(FileExistsError):
            start_run(self.models, "resnet18-y")

    def test_reports_changed_and_missing_files(self) -> None:
        staging = start_run(self.models, "resnet18-x")
        (staging / "a.txt").write_bytes(b"alpha")
        (staging / "b.txt").write_bytes(b"beta")
        final = finish_run(staging, {})
        (final / "a.txt").write_bytes(b"changed")
        (final / "b.txt").unlink()

        problems = check_files(final, read_run(final))

        self.assertEqual(problems, [
            "a.txt: checksum differs from run.json",
            "b.txt: missing from the run folder",
        ])

    def test_csv_round_trip_uses_unix_line_endings(self) -> None:
        path = self.models / "rows.csv"
        self.models.mkdir()

        write_csv(path, ("name", "value"), [{"name": "a", "value": 1}, {"name": "b", "value": 0.5}])

        self.assertNotIn(b"\r", path.read_bytes())
        self.assertEqual(read_csv(path), [{"name": "a", "value": "1"}, {"name": "b", "value": "0.5"}])


if __name__ == "__main__":
    unittest.main()
```

`tests/training/test_report.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from frog_classifier.training.report import CurveInput, render_report, write_precision_recall_figure
from tests.training.helpers import example_metadata


class ReportTests(unittest.TestCase):
    def test_report_states_results_threshold_and_errors(self) -> None:
        report = render_report(example_metadata(cross_validation=True))

        for expected in (
            "# Training run resnet18-20261014T000000Z",
            "| ResNet-18 | 0.900 | 0.900 | 0.900 | 0.900 | 0.600 |",
            "Threshold rule: maximum F1 on the validation fold, ties to the higher threshold.",
            "about 28.8 false detections per hour",
            "| Background | 48 | 2 |",
            "| Frog | 2 | 18 |",
            "| 01:00 | 30 | 2 | 15 | 1 |",
            "| unknown | 20 | 0 | 5 | 1 |",
            "| Test | 2 | 2 | 18 | 16 |",
            "## Cross-validation",
            "| ResNet-18 | 0.900 (0.020) |",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, report)
        self.assertNotIn("\u2014", report)

    def test_cross_validation_section_is_absent_when_skipped(self) -> None:
        report = render_report(example_metadata(cross_validation=False))

        self.assertNotIn("## Cross-validation", report)
        self.assertIn("Cross-validation was skipped", report)

    def test_writes_a_png_figure(self) -> None:
        labels = np.array([0, 0, 1, 1, 0, 1])
        panel = CurveInput(
            "Test", labels,
            resnet=np.array([0.1, 0.3, 0.8, 0.9, 0.2, 0.7]),
            baseline=np.array([0.4, 0.5, 0.6, 0.4, 0.3, 0.7]),
            resnet_threshold=0.7, baseline_threshold=0.5,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curves.png"

            write_precision_recall_figure(path, [panel, panel])

            self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
```

Check of the expected rows: `_split(20, 50, 18, 2)` gives precision 0.9, recall 0.9, F1 0.9, and the fixture's average precision is 0.9.

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python.exe -m unittest tests.training.test_artifact tests.training.test_report -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the run folder module**

`src/frog_classifier/training/artifact.py`:

```python
"""The saved run folder: staged creation, checksums, and reading back."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARTIFACT_SCHEMA_VERSION = 1
RUN_METADATA = "run.json"
MODEL_FILE = "model.pt"
PREDICTIONS_FILE = "predictions.csv"
PREDICTION_COLUMNS = (
    "example_id", "image_path", "image_sha256", "recording_id", "fold", "split",
    "label", "faint", "probability", "baseline_probability",
)
_STAGING_SUFFIX = ".partial"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def start_run(models_root: Path, name: str) -> Path:
    """Create the staging folder for a new run. Refuses a name already used or staged,
    so a run never overwrites another and an interrupted run never looks finished."""
    final = models_root / name
    staging = models_root / f".{name}{_STAGING_SUFFIX}"
    for path in (final, staging):
        if path.exists():
            raise FileExistsError(f"run folder already exists: {path}")
    staging.mkdir(parents=True)
    return staging


def finish_run(staging: Path, metadata: Mapping[str, Any]) -> Path:
    """Checksum every file, write run.json, and move the folder to its final name."""
    if not (staging.name.startswith(".") and staging.name.endswith(_STAGING_SUFFIX)):
        raise ValueError(f"not a staging folder: {staging}")
    files = {
        path.name: file_sha256(path)
        for path in sorted(staging.iterdir())
        if path.is_file() and path.name != RUN_METADATA
    }
    write_json(staging / RUN_METADATA, {**metadata, "files": files})
    final = staging.with_name(staging.name[1 : -len(_STAGING_SUFFIX)])
    staging.rename(final)
    return final


def read_run(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / RUN_METADATA).read_text(encoding="utf-8"))


def check_files(run_dir: Path, metadata: Mapping[str, Any]) -> list[str]:
    problems = []
    for name, expected in sorted(metadata["files"].items()):
        path = run_dir / name
        if not path.is_file():
            problems.append(f"{name}: missing from the run folder")
        elif file_sha256(path) != expected:
            problems.append(f"{name}: checksum differs from run.json")
    return problems


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_bytes((json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
```

- [ ] **Step 4: Write the report module**

`src/frog_classifier/training/report.py`:

```python
"""Human-readable run report and precision-recall figure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve

from .metrics import evaluate


@dataclass(frozen=True)
class CurveInput:
    title: str
    labels: np.ndarray
    resnet: np.ndarray
    baseline: np.ndarray
    resnet_threshold: float
    baseline_threshold: float


def write_precision_recall_figure(path: Path, panels: Sequence[CurveInput]) -> None:
    """One panel per split, both models' curves, and a dot at each chosen threshold."""
    figure = Figure(figsize=(5 * len(panels), 4.5), layout="constrained")
    for axes, panel in zip(figure.subplots(1, len(panels), squeeze=False)[0], panels):
        for name, probabilities, threshold, colour in (
            ("ResNet-18", panel.resnet, panel.resnet_threshold, "tab:blue"),
            ("Logistic baseline", panel.baseline, panel.baseline_threshold, "tab:gray"),
        ):
            precision, recall, _ = precision_recall_curve(panel.labels, probabilities)
            axes.plot(recall, precision, color=colour, label=name)
            point = evaluate(panel.labels, probabilities, threshold)
            axes.plot(point.recall, point.precision, "o", color=colour)
        axes.set(title=panel.title, xlabel="Recall", ylabel="Precision", xlim=(0, 1.01), ylim=(0, 1.01))
        axes.grid(alpha=0.3)
        axes.legend(loc="lower left")
    figure.savefig(path, dpi=120)


def render_report(metadata: Mapping[str, Any]) -> str:
    """Markdown summary built only from run.json content, so it can be regenerated."""
    metrics = metadata["metrics"]
    resnet, baseline = metrics["resnet"], metrics["baseline"]
    threshold = metadata["threshold"]["value"]
    test = resnet["test"]
    training = metadata["training"]
    data = metadata["data"]
    lines = [
        f"# Training run {metadata['run_name']}",
        "",
        "## Test results",
        "",
        "The test fold holds recordings that played no part in training or in choosing the threshold.",
        "",
        *_model_table([
            ("ResNet-18", resnet["test"], threshold),
            ("Logistic baseline", baseline["test"], baseline["threshold"]),
        ]),
        "",
        f"Threshold rule: {metadata['threshold']['rule']}.",
        f"At this threshold the ResNet flags {test['false_positive_rate']:.1%} of test background clips, "
        f"about {metrics['false_detections_per_hour']:.1f} false detections per hour of recording without a frog call.",
        "",
        "## Test confusion matrix",
        "",
        "| Actual | Predicted background | Predicted frog |",
        "| --- | ---: | ---: |",
        f"| Background | {test['true_negatives']} | {test['false_positives']} |",
        f"| Frog | {test['false_negatives']} | {test['true_positives']} |",
        "",
        "## Validation results",
        "",
        *_model_table([
            ("ResNet-18", resnet["validation"], threshold),
            ("Logistic baseline", baseline["validation"], baseline["threshold"]),
        ]),
        "",
        "## Test errors by recording hour",
        "",
        "| Hour | Background clips | False positives | Frog clips | Missed calls |",
        "| --- | ---: | ---: | ---: | ---: |",
        *(
            f"| {_hour(row['hour'])} | {row['background']} | {row['false_positives']} | {row['frog']} | {row['missed']} |"
            for row in metrics["test_by_hour"]
        ),
        "",
        "## Faint calls",
        "",
        "Frog clips labeled with the Frog faint button are counted separately from the other frog clips.",
        "",
        "| Split | Faint frog clips | Detected | Other frog clips | Detected |",
        "| --- | ---: | ---: | ---: | ---: |",
        *(
            f"| {split.capitalize()} | {counts['faint']['clips']} | {counts['faint']['detected']} "
            f"| {counts['other']['clips']} | {counts['other']['detected']} |"
            for split, counts in metrics["faint_recall"].items()
        ),
        "",
        *_cross_validation_section(metadata["cross_validation"]),
        "## Training",
        "",
        f"- Best validation epoch: {training['best_epoch']} of {training['epochs_run']} run.",
        f"- Device: {training['device_name']} ({training['device']}).",
        f"- Training time for all rotations: {training['seconds']:.0f} seconds.",
        f"- Seed: {metadata['seed']}.",
        f"- Starting weights: {metadata['model']['pretrained_weights'] or 'random initialization'}.",
        "",
        "## Data",
        "",
        "| Split | Clips | Frog | Background | Recordings |",
        "| --- | ---: | ---: | ---: | ---: |",
        *(
            f"| {name.capitalize()} | {split['examples']} | {split['frog']} | {split['background']} | {split['recordings']} |"
            for name, split in data["splits"].items()
        ),
        "",
        f"Test fold {data['test_fold']}, validation fold {data['validation_fold']}.",
        f"Faint-tagged frog clips in the manifest: {data['faint_tagged_frog_clips']}.",
        "",
        "## Provenance",
        "",
        "| Item | Value |",
        "| --- | --- |",
        *(f"| {name} SHA-256 | `{value}` |" for name, value in metadata["checksums"].items()),
        f"| Git commit | `{metadata['source']['git_commit']}` |",
        f"| Uncommitted changes | {metadata['source']['git_worktree_dirty']} |",
        *(f"| {name} | {value} |" for name, value in metadata["environment"].items()),
        "",
    ]
    return "\n".join(lines)


def _model_table(rows: Sequence[tuple[str, Mapping[str, Any], float]]) -> list[str]:
    return [
        "| Model | Average precision | Precision | Recall | F1 | Threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *(
            f"| {name} | {split['average_precision']:.3f} | {split['precision']:.3f} "
            f"| {split['recall']:.3f} | {split['f1']:.3f} | {threshold:.3f} |"
            for name, split, threshold in rows
        ),
    ]


def _cross_validation_section(cross_validation: Mapping[str, Any] | None) -> list[str]:
    if cross_validation is None:
        return ["Cross-validation was skipped for this run.", ""]
    lines = [
        "## Cross-validation",
        "",
        "Each rotation trains a new model with a different test fold and chooses its own threshold on the next fold.",
        "",
        "| Rotation | Test fold | Validation fold | ResNet AP | ResNet F1 | Baseline AP | Baseline F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *(
            f"| {row['rotation']} | {row['test_fold']} | {row['validation_fold']} "
            f"| {row['resnet']['average_precision']:.3f} | {row['resnet']['f1']:.3f} "
            f"| {row['baseline']['average_precision']:.3f} | {row['baseline']['f1']:.3f} |"
            for row in cross_validation["rotations"]
        ),
        "",
        "Mean and standard deviation over the rotations:",
        "",
        "| Model | Average precision | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, name in (("resnet", "ResNet-18"), ("baseline", "Logistic baseline")):
        summary = cross_validation["summary"][key]
        cells = " | ".join(
            f"{summary[metric]['mean']:.3f} ({summary[metric]['std']:.3f})"
            for metric in ("average_precision", "precision", "recall", "f1")
        )
        lines.append(f"| {name} | {cells} |")
    return [*lines, ""]


def _hour(hour: int | None) -> str:
    return "unknown" if hour is None else f"{hour:02d}:00"
```

The cross-validation assertion `"| ResNet-18 | 0.900 (0.020) |"` in the test matches the start of the ResNet summary row.

- [ ] **Step 5: Run them to see them pass**

Run: `.venv/Scripts/python.exe -m unittest tests.training.test_artifact tests.training.test_report -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/frog_classifier/training/artifact.py src/frog_classifier/training/report.py tests/training
git commit -m "feat: save training runs with checksums and a readable report"
```

---

### Task 8: Run orchestration and the training command

**Files:**
- Create: `src/frog_classifier/training/run.py`
- Create: `scripts/train_model.py`
- Modify: `tests/training/helpers.py` (add `build_synthetic_repo`)
- Test: `tests/training/test_run.py`

**Interfaces:**
- Consumes: everything from Tasks 2 to 7; `load_preprocessing_config`, `load_manifest`, `ManifestRow`, `parse_example_filename`, `DecisionLog`, `LOG_NAME`, `latest_decisions`.
- Produces: `RunPaths(repo_root, manifest, preprocessing_config, training_config, labeled_root, models_root)` with `RunPaths.defaults(repo_root)`; `train_and_save(paths: RunPaths, *, cross_validate: bool = True, device: torch.device | None = None, progress: Callable[[str], None] = print, clock: Callable[[], datetime] = utc_now) -> Path`; test helper `build_synthetic_repo(repo_root: Path, *, test_fold: int = 0, val_fold: int = 1, faint_log: bool = True) -> RunPaths`.

- [ ] **Step 1: Write the synthetic repository helper**

Append to `tests/training/helpers.py` (add the imports at the top of the file):

```python
from pathlib import Path

import numpy as np
from PIL import Image

from frog_classifier.data.manifest import ManifestRow, serialize_manifest
from frog_classifier.labeling.decisions import LOG_NAME, DecisionLog
from frog_classifier.training.run import RunPaths
from tests.data.helpers import load_test_config


IMAGE_SHAPE = (128, 216)


def build_synthetic_repo(
    repo_root: Path,
    *,
    test_fold: int = 0,
    val_fold: int = 1,
    faint_log: bool = True,
) -> RunPaths:
    """Five folds, each with one frog and one background recording of two clips.

    Frog clips carry a bright band where a call would sit; the first frog clip of
    the test fold is tagged faint in the decision log when faint_log is true.
    """
    config = load_test_config(repo_root)
    training_path = repo_root / "config" / "training.toml"
    training_path.write_bytes(TRAINING_CONFIG)
    labeled = repo_root / "labeled"
    rng = np.random.default_rng(0)
    rows = []
    for fold in range(5):
        split = "test" if fold == test_fold else "val" if fold == val_fold else "train"
        for label_name, label, hour in (("litoria_aurea", 1, 2), ("non_target", 0, 22)):
            recording_id = f"2022010{fold + 1}_{hour:02d}0000"
            for start_s in (0, 5):
                example_id = f"{recording_id}_start{start_s}s"
                image_path = f"labeled/{label_name}/{example_id}.png"
                pixels = rng.integers(0, 40, IMAGE_SHAPE, dtype=np.uint8)
                if label:
                    pixels[60:70, :] = 220
                (repo_root / image_path).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(repo_root / image_path)
                rows.append(ManifestRow(
                    manifest_version=1,
                    example_id=example_id,
                    image_path=image_path,
                    label=label,
                    label_name=label_name,
                    recording_id=recording_id,
                    start_s=start_s,
                    fold=fold,
                    split=split,
                    preprocessing_config_sha256=config.sha256,
                ))
    manifest = labeled / "manifest.csv"
    manifest.write_bytes(serialize_manifest(tuple(rows)))
    if faint_log:
        faint_clip = next(row for row in rows if row.fold == test_fold and row.label == 1)
        DecisionLog(labeled / LOG_NAME).append(faint_clip.example_id, "litoria_aurea", "label", faint=True)
    return RunPaths(
        repo_root=repo_root,
        manifest=manifest,
        preprocessing_config=repo_root / "config" / "preprocessing.toml",
        training_config=training_path,
        labeled_root=labeled,
        models_root=repo_root / "models",
    )
```

- [ ] **Step 2: Write the failing tests**

`tests/training/test_run.py`:

```python
from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from frog_classifier.training.artifact import check_files, read_csv, read_run
from frog_classifier.training.run import train_and_save
from tests.training.helpers import build_synthetic_repo


CPU = torch.device("cpu")
FIXED_TIME = datetime(2026, 10, 14, 1, 2, 3, tzinfo=timezone.utc)


def _quiet(_: str) -> None:
    pass


class CrossValidatedRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.paths = build_synthetic_repo(Path(cls.directory.name))
        cls.messages: list[str] = []
        cls.run_dir = train_and_save(cls.paths, device=CPU, progress=cls.messages.append, clock=lambda: FIXED_TIME)
        cls.metadata = read_run(cls.run_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_writes_every_file_under_the_final_name_only(self) -> None:
        self.assertEqual(self.run_dir.name, "resnet18-20261014T010203Z")
        self.assertEqual(
            sorted(path.name for path in self.run_dir.iterdir()),
            sorted([
                "cross_validation.csv", "history.csv", "manifest.csv", "model.pt",
                "out_of_fold_predictions.csv", "precision_recall.png", "predictions.csv",
                "preprocessing.toml", "report.md", "run.json", "training.toml",
            ]),
        )
        self.assertEqual([path.name for path in self.paths.models_root.iterdir()], [self.run_dir.name])
        self.assertEqual(check_files(self.run_dir, self.metadata), [])

    def test_copies_inputs_and_records_their_checksums(self) -> None:
        self.assertEqual((self.run_dir / "manifest.csv").read_bytes(), self.paths.manifest.read_bytes())
        self.assertEqual(
            self.metadata["checksums"]["manifest"],
            hashlib.sha256(self.paths.manifest.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            (self.run_dir / "training.toml").read_bytes(), self.paths.training_config.read_bytes()
        )

    def test_predictions_cover_every_example_with_its_image_checksum(self) -> None:
        rows = read_csv(self.run_dir / "predictions.csv")

        self.assertEqual(len(rows), 20)
        for row in rows:
            image = self.paths.repo_root / row["image_path"]
            self.assertEqual(row["image_sha256"], hashlib.sha256(image.read_bytes()).hexdigest())
            self.assertTrue(0.0 <= float(row["probability"]) <= 1.0)
            self.assertTrue(0.0 <= float(row["baseline_probability"]) <= 1.0)
        self.assertEqual(sorted(row["faint"] for row in rows if row["label"] == "1"), ["0"] * 9 + ["1"])
        self.assertTrue(all(row["faint"] == "" for row in rows if row["label"] == "0"))

    def test_every_fold_is_tested_once_and_every_example_gets_one_out_of_fold_probability(self) -> None:
        rotations = self.metadata["cross_validation"]["rotations"]
        self.assertEqual(sorted(row["test_fold"] for row in rotations), [0, 1, 2, 3, 4])
        self.assertEqual(rotations[0]["test_fold"], self.metadata["data"]["test_fold"])
        out_of_fold = read_csv(self.run_dir / "out_of_fold_predictions.csv")
        self.assertEqual(len(out_of_fold), 20)
        self.assertEqual(len({row["example_id"] for row in out_of_fold}), 20)
        self.assertEqual(len(read_csv(self.run_dir / "cross_validation.csv")), 10)

    def test_metadata_records_the_run(self) -> None:
        self.assertEqual(self.metadata["data"]["faint_tagged_frog_clips"], 1)
        self.assertEqual(self.metadata["metrics"]["faint_recall"]["test"]["faint"]["clips"], 1)
        self.assertEqual(self.metadata["training"]["device"], "cpu")
        self.assertIsNone(self.metadata["model"]["pretrained_weights"])
        self.assertEqual(self.metadata["model"]["input_shape"], [1, 128, 216])
        self.assertIsNone(self.metadata["source"]["git_commit"])
        self.assertEqual(self.metadata["data"]["splits"]["test"]["recordings"], 2)
        self.assertTrue(0.0 <= self.metadata["threshold"]["value"] <= 1.0)
        self.assertIn("Cross-validation", (self.run_dir / "report.md").read_text(encoding="utf-8"))
        self.assertTrue(any(message.startswith("epoch") for message in self.messages))


class SingleRunTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_skipping_cross_validation_trains_once_and_writes_no_rotation_files(self) -> None:
        paths = build_synthetic_repo(self.root)

        run_dir = train_and_save(paths, cross_validate=False, device=CPU, progress=_quiet)

        self.assertIsNone(read_run(run_dir)["cross_validation"])
        self.assertFalse((run_dir / "cross_validation.csv").exists())
        self.assertFalse((run_dir / "out_of_fold_predictions.csv").exists())

    def test_rotation_zero_follows_a_manifest_with_other_folds(self) -> None:
        paths = build_synthetic_repo(self.root, test_fold=2, val_fold=3)

        metadata = read_run(train_and_save(paths, cross_validate=False, device=CPU, progress=_quiet))

        self.assertEqual((metadata["data"]["test_fold"], metadata["data"]["validation_fold"]), (2, 3))

    def test_trains_without_a_decision_log(self) -> None:
        paths = build_synthetic_repo(self.root, faint_log=False)

        metadata = read_run(train_and_save(paths, cross_validate=False, device=CPU, progress=_quiet))

        self.assertEqual(metadata["data"]["faint_tagged_frog_clips"], 0)

    def test_refuses_an_existing_run_folder_before_training(self) -> None:
        paths = build_synthetic_repo(self.root)
        (paths.models_root / "resnet18-20261014T010203Z").mkdir(parents=True)
        messages: list[str] = []

        with self.assertRaises(FileExistsError):
            train_and_save(paths, device=CPU, progress=messages.append, clock=lambda: FIXED_TIME)

        self.assertFalse(any(message.startswith("epoch") for message in messages))

    def test_a_failed_run_leaves_no_folder_under_a_final_name(self) -> None:
        paths = build_synthetic_repo(self.root)
        first_image = sorted((paths.labeled_root / "non_target").glob("*.png"))[0]
        Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(first_image)

        with self.assertRaisesRegex(ValueError, first_image.name):
            train_and_save(paths, device=CPU, progress=_quiet, clock=lambda: FIXED_TIME)

        self.assertEqual(
            [path.name for path in paths.models_root.iterdir()],
            [".resnet18-20261014T010203Z.partial"],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run them to see them fail**

Run: `.venv/Scripts/python.exe -m unittest tests.training.test_run -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'frog_classifier.training.run'`.

- [ ] **Step 4: Write the orchestration**

`src/frog_classifier/training/run.py`:

```python
"""One complete training run, from the validated manifest to a saved run folder."""

from __future__ import annotations

import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torchvision
from PIL import Image

from frog_classifier.data.config import PreprocessingConfig, load_preprocessing_config
from frog_classifier.data.manifest import ManifestRow, load_manifest
from frog_classifier.data.naming import parse_example_filename
from frog_classifier.labeling.decisions import LOG_NAME, DecisionLog, latest_decisions

from .artifact import (
    ARTIFACT_SCHEMA_VERSION,
    MODEL_FILE,
    PREDICTION_COLUMNS,
    PREDICTIONS_FILE,
    file_sha256,
    finish_run,
    start_run,
    write_csv,
)
from .baseline import baseline_features, fit_baseline
from .config import TrainingConfig, load_training_config
from .fit import FitResult, fit_model, predict_probabilities, select_device
from .images import expected_image_shape, load_spectrograms
from .metrics import (
    THRESHOLD_RULE,
    counts_by_hour,
    evaluate,
    false_detections_per_hour,
    recall_by_faint,
    select_threshold,
)
from .model import ARCHITECTURE, PRETRAINED_WEIGHTS
from .report import CurveInput, render_report, write_precision_recall_figure


HISTORY_COLUMNS = ("epoch", "train_loss", "val_loss", "val_average_precision", "learning_rate", "seconds")
CROSS_VALIDATION_COLUMNS = (
    "rotation", "test_fold", "validation_fold", "best_epoch", "model", "threshold",
    "average_precision", "precision", "recall", "f1", "false_positive_rate",
)
OUT_OF_FOLD_COLUMNS = ("example_id", "image_path", "recording_id", "fold", "label", "faint", "probability", "threshold")


@dataclass(frozen=True)
class RunPaths:
    repo_root: Path
    manifest: Path
    preprocessing_config: Path
    training_config: Path
    labeled_root: Path
    models_root: Path

    @classmethod
    def defaults(cls, repo_root: Path) -> RunPaths:
        return cls(
            repo_root=repo_root,
            manifest=repo_root / "labeled" / "manifest.csv",
            preprocessing_config=repo_root / "config" / "preprocessing.toml",
            training_config=repo_root / "config" / "training.toml",
            labeled_root=repo_root / "labeled",
            models_root=repo_root / "models",
        )


@dataclass(frozen=True)
class _Dataset:
    rows: tuple[ManifestRow, ...]
    pixels: np.ndarray
    sha256: tuple[str, ...]
    labels: np.ndarray
    folds: np.ndarray
    faint: np.ndarray
    baseline_features: np.ndarray


@dataclass(frozen=True)
class _Rotation:
    test_fold: int
    val_fold: int
    fit: FitResult
    resnet: np.ndarray
    baseline: np.ndarray
    resnet_threshold: float
    baseline_threshold: float


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def train_and_save(
    paths: RunPaths,
    *,
    cross_validate: bool = True,
    device: torch.device | None = None,
    progress: Callable[[str], None] = print,
    clock: Callable[[], datetime] = utc_now,
) -> Path:
    """Train, evaluate, and save one run; return its folder.

    Rotation zero uses the manifest's own test and validation folds and is the
    saved model. Cross-validation adds one rotation per remaining fold.
    """
    preprocessing = load_preprocessing_config(paths.preprocessing_config)
    training = load_training_config(paths.training_config)
    rows = load_manifest(paths.manifest, paths.repo_root, preprocessing, label_root=paths.labeled_root)
    test_fold, val_fold = _manifest_folds(rows)
    created_at = clock()
    name = f"{ARCHITECTURE}-{created_at:%Y%m%dT%H%M%SZ}"
    staging = start_run(paths.models_root, name)
    device = device if device is not None else select_device()

    progress(f"Loading {len(rows)} images")
    data = _load_dataset(rows, paths, preprocessing, training)
    fold_ids = sorted(set(data.folds.tolist()))
    count = len(fold_ids) if cross_validate else 1
    rotations: list[_Rotation] = []
    started = time.perf_counter()
    for offset in range(count):
        rotation_test = fold_ids[(fold_ids.index(test_fold) + offset) % len(fold_ids)]
        rotation_val = fold_ids[(fold_ids.index(val_fold) + offset) % len(fold_ids)]
        progress(f"Rotation {offset + 1} of {count}: test fold {rotation_test}, validation fold {rotation_val}, device {device}")
        rotations.append(_train_rotation(data, rotation_test, rotation_val, training, preprocessing, device, progress))
    seconds = time.perf_counter() - started

    metadata = _metadata(
        name, created_at, data, rotations, training, preprocessing, paths, device, seconds, cross_validate,
    )
    _write_files(staging, data, rotations, training, preprocessing, paths, metadata, cross_validate)
    return finish_run(staging, metadata)


def _manifest_folds(rows: tuple[ManifestRow, ...]) -> tuple[int, int]:
    """The single test fold and single validation fold the manifest's split column names."""
    folds_by_split: dict[str, set[int]] = {}
    for row in rows:
        folds_by_split.setdefault(row.split, set()).add(row.fold)
    test, val = folds_by_split.get("test", set()), folds_by_split.get("val", set())
    if len(test) != 1 or len(val) != 1 or test & folds_by_split.get("train", set()) or val & folds_by_split.get("train", set()):
        raise ValueError("the manifest must hold exactly one test fold and one validation fold, apart from training folds")
    return next(iter(test)), next(iter(val))


def _load_dataset(
    rows: tuple[ManifestRow, ...],
    paths: RunPaths,
    preprocessing: PreprocessingConfig,
    training: TrainingConfig,
) -> _Dataset:
    loaded = load_spectrograms([paths.repo_root / row.image_path for row in rows], expected_image_shape(preprocessing))
    faint_ids = _faint_example_ids(paths.labeled_root, preprocessing.audio.chunk_seconds)
    return _Dataset(
        rows=rows,
        pixels=loaded.pixels,
        sha256=loaded.sha256,
        labels=np.array([row.label for row in rows], dtype=np.int64),
        folds=np.array([row.fold for row in rows], dtype=np.int64),
        faint=np.array([row.label == 1 and row.example_id in faint_ids for row in rows], dtype=bool),
        baseline_features=baseline_features(
            (Image.fromarray(image) for image in loaded.pixels), size=training.baseline.image_size,
        ),
    )


def _faint_example_ids(labeled_root: Path, chunk_seconds: int) -> frozenset[str]:
    """Frog clips whose latest frog decision used the Frog faint button, per the decision log."""
    clips = latest_decisions(labeled_root, DecisionLog(labeled_root / LOG_NAME), chunk_seconds=chunk_seconds)
    return frozenset(
        parse_example_filename(clip.path, chunk_seconds=chunk_seconds).example_id
        for clip in clips
        if clip.folder == "litoria_aurea" and clip.faint
    )


def _train_rotation(
    data: _Dataset,
    test_fold: int,
    val_fold: int,
    training: TrainingConfig,
    preprocessing: PreprocessingConfig,
    device: torch.device,
    progress: Callable[[str], None],
) -> _Rotation:
    train = (data.folds != test_fold) & (data.folds != val_fold)
    val = data.folds == val_fold
    fit = fit_model(
        data.pixels[train], data.labels[train], data.pixels[val], data.labels[val], training,
        db_floor=preprocessing.normalization.db_floor,
        db_ceiling=preprocessing.normalization.db_ceiling,
        device=device,
        progress=progress,
    )
    resnet = predict_probabilities(fit.model, data.pixels, device)
    classifier = fit_baseline(
        data.baseline_features[train], data.labels[train],
        inverse_regularization=training.baseline.inverse_regularization,
    )
    baseline = classifier.predict_proba(data.baseline_features)[:, list(classifier.classes_).index(1)]
    return _Rotation(
        test_fold=test_fold,
        val_fold=val_fold,
        fit=fit,
        resnet=resnet,
        baseline=baseline,
        resnet_threshold=select_threshold(data.labels[val], resnet[val]),
        baseline_threshold=select_threshold(data.labels[val], baseline[val]),
    )


def _metadata(
    name: str,
    created_at: datetime,
    data: _Dataset,
    rotations: list[_Rotation],
    training: TrainingConfig,
    preprocessing: PreprocessingConfig,
    paths: RunPaths,
    device: torch.device,
    seconds: float,
    cross_validate: bool,
) -> dict[str, Any]:
    primary = rotations[0]
    labels = data.labels
    masks = {
        "train": (data.folds != primary.test_fold) & (data.folds != primary.val_fold),
        "validation": data.folds == primary.val_fold,
        "test": data.folds == primary.test_fold,
    }
    test = masks["test"]
    resnet_test = evaluate(labels[test], primary.resnet[test], primary.resnet_threshold)
    recording_ids = np.array([row.recording_id for row in data.rows])
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_name": name,
        "created_at": created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "architecture": ARCHITECTURE,
            "pretrained_weights": PRETRAINED_WEIGHTS if training.pretrained else None,
            "input_shape": [1, *data.pixels.shape[1:]],
            "classes": dict(preprocessing.classes),
            "positive_class": "litoria_aurea",
            "output": "one logit per image; the frog probability is its sigmoid",
        },
        "threshold": {"value": primary.resnet_threshold, "rule": THRESHOLD_RULE},
        "seed": training.seed,
        "training": {
            "best_epoch": primary.fit.best_epoch,
            "epochs_run": len(primary.fit.history),
            "device": device.type,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "seconds": round(seconds, 1),
        },
        "data": {
            "examples": len(data.rows),
            "test_fold": primary.test_fold,
            "validation_fold": primary.val_fold,
            "faint_tagged_frog_clips": int(data.faint.sum()),
            "splits": {
                split: {
                    "examples": int(mask.sum()),
                    "frog": int(labels[mask].sum()),
                    "background": int((labels[mask] == 0).sum()),
                    "recordings": len(set(recording_ids[mask].tolist())),
                }
                for split, mask in masks.items()
            },
        },
        "checksums": {
            "manifest": file_sha256(paths.manifest),
            "preprocessing_config": preprocessing.sha256,
            "training_config": training.sha256,
            "uv_lock": file_sha256(paths.repo_root / "uv.lock") if (paths.repo_root / "uv.lock").is_file() else None,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "platform": platform.platform(),
        },
        "source": _git_state(paths.repo_root),
        "metrics": {
            "resnet": {
                "validation": asdict(evaluate(labels[masks["validation"]], primary.resnet[masks["validation"]], primary.resnet_threshold)),
                "test": asdict(resnet_test),
            },
            "baseline": {
                "threshold": primary.baseline_threshold,
                "validation": asdict(evaluate(labels[masks["validation"]], primary.baseline[masks["validation"]], primary.baseline_threshold)),
                "test": asdict(evaluate(labels[test], primary.baseline[test], primary.baseline_threshold)),
            },
            "false_detections_per_hour": false_detections_per_hour(
                resnet_test.false_positive_rate, preprocessing.audio.chunk_seconds,
            ),
            "test_by_hour": [
                asdict(row)
                for row in counts_by_hour(recording_ids[test].tolist(), labels[test], primary.resnet[test], primary.resnet_threshold)
            ],
            "faint_recall": {
                split: {
                    group: asdict(count)
                    for group, count in recall_by_faint(
                        labels[masks[split]], data.faint[masks[split]], primary.resnet[masks[split]], primary.resnet_threshold,
                    ).items()
                }
                for split in ("validation", "test")
            },
        },
        "cross_validation": _cross_validation(data, rotations) if cross_validate else None,
    }


def _cross_validation(data: _Dataset, rotations: list[_Rotation]) -> dict[str, Any]:
    per_rotation = []
    for index, rotation in enumerate(rotations):
        test = data.folds == rotation.test_fold
        per_rotation.append({
            "rotation": index,
            "test_fold": rotation.test_fold,
            "validation_fold": rotation.val_fold,
            "best_epoch": rotation.fit.best_epoch,
            "resnet_threshold": rotation.resnet_threshold,
            "baseline_threshold": rotation.baseline_threshold,
            "resnet": asdict(evaluate(data.labels[test], rotation.resnet[test], rotation.resnet_threshold)),
            "baseline": asdict(evaluate(data.labels[test], rotation.baseline[test], rotation.baseline_threshold)),
        })
    summary = {
        model: {
            metric: _spread([row[model][metric] for row in per_rotation])
            for metric in ("average_precision", "precision", "recall", "f1")
        }
        for model in ("resnet", "baseline")
    }
    return {"rotations": per_rotation, "summary": summary}


def _spread(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _git_state(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_worktree_dirty": None}
    return {"git_commit": commit, "git_worktree_dirty": bool(status.strip())}


def _write_files(
    staging: Path,
    data: _Dataset,
    rotations: list[_Rotation],
    training: TrainingConfig,
    preprocessing: PreprocessingConfig,
    paths: RunPaths,
    metadata: dict[str, Any],
    cross_validate: bool,
) -> None:
    primary = rotations[0]
    torch.save({key: value.cpu() for key, value in primary.fit.model.state_dict().items()}, staging / MODEL_FILE)
    shutil.copyfile(preprocessing.source_path, staging / "preprocessing.toml")
    shutil.copyfile(training.source_path, staging / "training.toml")
    shutil.copyfile(paths.manifest, staging / "manifest.csv")
    write_csv(staging / PREDICTIONS_FILE, PREDICTION_COLUMNS, (
        {
            "example_id": row.example_id,
            "image_path": row.image_path,
            "image_sha256": checksum,
            "recording_id": row.recording_id,
            "fold": row.fold,
            "split": row.split,
            "label": row.label,
            "faint": int(faint) if row.label == 1 else "",
            "probability": f"{resnet:.8f}",
            "baseline_probability": f"{baseline:.8f}",
        }
        for row, checksum, faint, resnet, baseline in zip(
            data.rows, data.sha256, data.faint, primary.resnet, primary.baseline, strict=True,
        )
    ))
    write_csv(staging / "history.csv", HISTORY_COLUMNS, (asdict(record) for record in primary.fit.history))
    if cross_validate:
        _write_cross_validation(staging, data, rotations, metadata)
    labels = data.labels
    write_precision_recall_figure(staging / "precision_recall.png", [
        CurveInput(
            title, labels[mask], primary.resnet[mask], primary.baseline[mask],
            primary.resnet_threshold, primary.baseline_threshold,
        )
        for title, mask in (
            ("Validation", data.folds == primary.val_fold),
            ("Test", data.folds == primary.test_fold),
        )
    ])
    (staging / "report.md").write_bytes(render_report(metadata).encode("utf-8"))


def _write_cross_validation(
    staging: Path,
    data: _Dataset,
    rotations: list[_Rotation],
    metadata: dict[str, Any],
) -> None:
    write_csv(staging / "cross_validation.csv", CROSS_VALIDATION_COLUMNS, (
        {
            "rotation": row["rotation"],
            "test_fold": row["test_fold"],
            "validation_fold": row["validation_fold"],
            "best_epoch": row["best_epoch"],
            "model": model,
            "threshold": row[f"{model}_threshold"],
            **{metric: row[model][metric] for metric in ("average_precision", "precision", "recall", "f1", "false_positive_rate")},
        }
        for row in metadata["cross_validation"]["rotations"]
        for model in ("resnet", "baseline")
    ))
    by_test_fold = {rotation.test_fold: rotation for rotation in rotations}
    write_csv(staging / "out_of_fold_predictions.csv", OUT_OF_FOLD_COLUMNS, (
        {
            "example_id": row.example_id,
            "image_path": row.image_path,
            "recording_id": row.recording_id,
            "fold": row.fold,
            "label": row.label,
            "faint": int(faint) if row.label == 1 else "",
            "probability": f"{by_test_fold[row.fold].resnet[index]:.8f}",
            "threshold": f"{by_test_fold[row.fold].resnet_threshold:.8f}",
        }
        for index, (row, faint) in enumerate(zip(data.rows, data.faint, strict=True))
    ))
```

- [ ] **Step 5: Write the command**

`scripts/train_model.py`:

```python
"""Train the ResNet-18 frog detector from the validated manifest and save a run folder under models/."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from frog_classifier.training.artifact import read_run
from frog_classifier.training.fit import select_device
from frog_classifier.training.run import RunPaths, train_and_save


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    defaults = RunPaths.defaults(repo_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=defaults.manifest)
    parser.add_argument("--training-config", type=Path, default=defaults.training_config)
    parser.add_argument("--models-root", type=Path, default=defaults.models_root)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None, help="Default: the GPU when available.")
    parser.add_argument(
        "--no-cross-validation",
        action="store_true",
        help="Train only the saved model and skip the four extra fold rotations.",
    )
    args = parser.parse_args()

    paths = replace(
        defaults,
        manifest=_under(repo_root, args.manifest),
        training_config=_under(repo_root, args.training_config),
        models_root=_under(repo_root, args.models_root),
    )
    run_dir = train_and_save(paths, cross_validate=not args.no_cross_validation, device=select_device(args.device))

    metadata = read_run(run_dir)
    test = metadata["metrics"]["resnet"]["test"]
    print(f"Saved run: {run_dir}")
    print(
        f"Test: average precision {test['average_precision']:.3f}, precision {test['precision']:.3f}, "
        f"recall {test['recall']:.3f}, F1 {test['f1']:.3f} at threshold {metadata['threshold']['value']:.3f}"
    )
    print(f"Report: {run_dir / 'report.md'}")
    return 0


def _under(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run the tests to see them pass**

Run: `.venv/Scripts/python.exe -m unittest tests.training.test_run -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/frog_classifier/training/run.py scripts/train_model.py tests/training
git commit -m "feat: train, evaluate, cross-validate, and save a ResNet run"
```

---

### Task 9: Verification and the command tests

**Files:**
- Create: `src/frog_classifier/training/verify.py`
- Create: `scripts/verify_model.py`
- Test: `tests/training/test_verify.py`, `tests/training/test_commands.py`

**Interfaces:**
- Consumes: Tasks 3, 5, 7, 8.
- Produces: `PROBABILITY_TOLERANCE = 1e-4`; `verify_run(run_dir: Path, repo_root: Path, *, device: torch.device) -> VerificationResult(checked: int, largest_difference: float, problems: tuple[str, ...])` with property `passed`.

- [ ] **Step 1: Write the failing tests**

`tests/training/test_verify.py`:

```python
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from frog_classifier.training.artifact import PREDICTIONS_FILE, read_csv
from frog_classifier.training.run import train_and_save
from frog_classifier.training.verify import PROBABILITY_TOLERANCE, verify_run
from tests.training.helpers import build_synthetic_repo


CPU = torch.device("cpu")


class VerifyTests(unittest.TestCase):
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

    def test_a_fresh_run_reproduces_its_predictions(self) -> None:
        result = verify_run(self.run_dir, self.repo, device=CPU)

        self.assertTrue(result.passed, result.problems)
        self.assertEqual(result.checked, 20)
        self.assertLessEqual(result.largest_difference, PROBABILITY_TOLERANCE)

    def test_a_changed_image_is_reported_as_changed(self) -> None:
        row = read_csv(self.run_dir / PREDICTIONS_FILE)[0]
        Image.fromarray(np.full((128, 216), 7, dtype=np.uint8)).save(self.repo / row["image_path"])

        result = verify_run(self.run_dir, self.repo, device=CPU)

        self.assertFalse(result.passed)
        self.assertEqual(result.problems, (f"{row['example_id']}: image changed since training",))
        self.assertEqual(result.checked, 19)

    def test_a_relabeled_clip_is_reported_as_missing(self) -> None:
        row = read_csv(self.run_dir / PREDICTIONS_FILE)[0]
        (self.repo / row["image_path"]).unlink()

        result = verify_run(self.run_dir, self.repo, device=CPU)

        self.assertEqual(result.problems, (f"{row['example_id']}: image missing at {row['image_path']}",))

    def test_a_tampered_run_file_fails_its_checksum(self) -> None:
        predictions = self.run_dir / PREDICTIONS_FILE
        predictions.write_bytes(predictions.read_bytes().replace(b"0.", b"1.", 1))

        result = verify_run(self.run_dir, self.repo, device=CPU)

        self.assertEqual(result.problems, ("predictions.csv: checksum differs from run.json",))
        self.assertEqual(result.checked, 0)


if __name__ == "__main__":
    unittest.main()
```

`tests/training/test_commands.py`:

```python
from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from frog_classifier.training.artifact import PREDICTIONS_FILE
from tests.training.helpers import build_synthetic_repo


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(script: str, repo_root: Path):
    name = f"{script}_under_test"
    specification = importlib.util.spec_from_file_location(name, SCRIPTS / f"{script}.py")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    # The commands find the repository from their own location.
    module.__file__ = str(repo_root / "scripts" / f"{script}.py")
    return module


def _run(module, *arguments: str) -> tuple[int, str]:
    output = io.StringIO()
    with patch.object(sys, "argv", [module.__name__, *arguments]), redirect_stdout(output):
        return module.main(), output.getvalue()


class CommandTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo = Path(directory.name)
        build_synthetic_repo(self.repo)

    def test_train_then_verify_from_the_command_line(self) -> None:
        code, output = _run(_load("train_model", self.repo), "--no-cross-validation", "--device", "cpu")

        self.assertEqual(code, 0)
        self.assertIn("Saved run:", output)
        (run_dir,) = (self.repo / "models").iterdir()
        self.assertFalse((run_dir / "cross_validation.csv").exists())

        verify = _load("verify_model", self.repo)
        code, output = _run(verify, str(run_dir.relative_to(self.repo)), "--device", "cpu")
        self.assertEqual(code, 0)
        self.assertIn("PASSED", output)

        predictions = run_dir / PREDICTIONS_FILE
        predictions.write_bytes(predictions.read_bytes() + b"\n")
        code, output = _run(verify, str(run_dir), "--device", "cpu")
        self.assertEqual(code, 1)
        self.assertIn("FAIL predictions.csv: checksum differs from run.json", output)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/Scripts/python.exe -m unittest tests.training.test_verify tests.training.test_commands -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'frog_classifier.training.verify'`.

- [ ] **Step 3: Write the verification module**

`src/frog_classifier/training/verify.py`:

```python
"""Prove that a saved run reproduces its own predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .artifact import MODEL_FILE, PREDICTIONS_FILE, check_files, file_sha256, read_csv, read_run
from .fit import predict_probabilities, use_full_precision
from .images import load_spectrograms
from .model import ARCHITECTURE, SpectrogramResNet


# Saved probabilities have eight decimals; GPU and CPU fp32 agree far more closely than this.
PROBABILITY_TOLERANCE = 1e-4


@dataclass(frozen=True)
class VerificationResult:
    checked: int
    largest_difference: float
    problems: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.problems


def verify_run(run_dir: Path, repo_root: Path, *, device: torch.device) -> VerificationResult:
    """Check the run's file checksums, then recompute every saved probability.

    Images are read from their manifest paths under repo_root. An image that is
    gone or whose bytes changed since training is reported as such and skipped.
    """
    metadata = read_run(run_dir)
    problems = check_files(run_dir, metadata)
    if problems:
        return VerificationResult(0, 0.0, tuple(problems))
    if metadata["model"]["architecture"] != ARCHITECTURE:
        return VerificationResult(0, 0.0, (f"unsupported architecture {metadata['model']['architecture']!r}",))

    use_full_precision()
    model = SpectrogramResNet(pretrained=False)
    model.load_state_dict(torch.load(run_dir / MODEL_FILE, map_location="cpu", weights_only=True))
    model.to(device)

    readable = []
    for row in read_csv(run_dir / PREDICTIONS_FILE):
        path = repo_root / row["image_path"]
        if not path.is_file():
            problems.append(f"{row['example_id']}: image missing at {row['image_path']}")
        elif file_sha256(path) != row["image_sha256"]:
            problems.append(f"{row['example_id']}: image changed since training")
        else:
            readable.append((row, path))
    if not readable:
        return VerificationResult(0, 0.0, tuple(problems))

    shape = tuple(metadata["model"]["input_shape"][1:])
    pixels = load_spectrograms([path for _, path in readable], shape).pixels
    recomputed = predict_probabilities(model, pixels, device)
    saved = np.array([float(row["probability"]) for row, _ in readable])
    differences = np.abs(recomputed - saved)
    for (row, _), new, old, difference in zip(readable, recomputed, saved, differences, strict=True):
        if difference > PROBABILITY_TOLERANCE:
            problems.append(f"{row['example_id']}: probability {new:.6f} differs from saved {old:.6f}")
    return VerificationResult(len(readable), float(differences.max()), tuple(problems))
```

- [ ] **Step 4: Write the command**

`scripts/verify_model.py`:

```python
"""Check that a saved training run reproduces its own predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from frog_classifier.training.fit import select_device
from frog_classifier.training.verify import verify_run


LISTED_PROBLEMS = 20


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run folder, for example models/resnet18-20261014T113000Z.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None, help="Default: the GPU when available.")
    args = parser.parse_args()

    run_dir = args.run if args.run.is_absolute() else repo_root / args.run
    result = verify_run(run_dir, repo_root, device=select_device(args.device))

    print(f"Checked {result.checked} predictions; largest difference {result.largest_difference:.2e}")
    for problem in result.problems[:LISTED_PROBLEMS]:
        print(f"FAIL {problem}")
    if len(result.problems) > LISTED_PROBLEMS:
        print(f"... and {len(result.problems) - LISTED_PROBLEMS} more")
    print("PASSED" if result.passed else f"FAILED with {len(result.problems)} problems")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `.venv/Scripts/python.exe -m unittest tests.training.test_verify tests.training.test_commands -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/frog_classifier/training/verify.py scripts/verify_model.py tests/training
git commit -m "feat: verify that a saved run reproduces its predictions"
```

---

### Task 10: Repository checks and documentation

**Files:**
- Modify: `tests/test_repository_reproducibility.py`
- Modify: `docs/workflow.md`, `docs/architecture.md`, `docs/outline.md`, `README.md`

**Interfaces:**
- Consumes: the commands from Tasks 8 and 9.

- [ ] **Step 1: Require the training files and document the commands in the tests**

In `tests/test_repository_reproducibility.py`:
- Add a `TRAINING_PACKAGE_MODULES` tuple listing the twelve files under `src/frog_classifier/training/` (`__init__`, `artifact`, `augment`, `baseline`, `config`, `fit`, `images`, `metrics`, `model`, `report`, `run`, `verify`) and include it with `"config/training.toml"` in `test_project_navigation_files_are_complete_and_linked`.
- Add `"scripts/train_model.py"` and `"scripts/verify_model.py"` to `OPERATIONAL_TEXT_FILES`.
- Add `"models/resnet18-20261014T000000Z/model.pt"` to `GENERATED_PATHS`.
- Add `"uv run python scripts/train_model.py"` and `"uv run python scripts/verify_model.py"` to the commands in `test_workflow_documents_every_operational_command`.

Run: `.venv/Scripts/python.exe -m unittest tests.test_repository_reproducibility`
Expected: FAIL on the workflow commands only.

- [ ] **Step 2: Update the documents**

- `docs/workflow.md`: in Environment setup, say `uv sync --frozen` also installs the `model` group with PyTorch from the CUDA 13.0 index, and change the export command to `--no-default-groups`. Add a section "Train and verify the detector" after the baseline section with the commands `uv run python scripts/train_model.py`, `uv run python scripts/train_model.py --no-cross-validation`, `uv run python scripts/verify_model.py models/<run folder>`, and `uv run python scripts/verify_model.py models/<run folder> --device cpu`, and a short description of the run folder, the threshold rule, and what verification proves. Add both new scripts to the `py_compile` line.
- `docs/architecture.md`: add a "Model training" section describing the package modules, the rotation scheme, the run folder table, and verification; add `config/training.toml`, `src/frog_classifier/training/`, and `models/<run>/` to the storage contract; extend the Mermaid pipeline with the manifest feeding "ResNet-18 training" and a saved run; mention the `model` dependency group in the environment contract.
- `docs/outline.md` and `README.md`: describe Phase 4's training command in beginner terms and add the two scripts to the repository tree.

Each sentence on its own line, no em dashes.

- [ ] **Step 3: Run the full suite and checks**

```bash
.venv/Scripts/python.exe -m unittest discover -s tests
.venv/Scripts/python.exe -B -m py_compile scripts/build_manifest.py scripts/label_frontend.py scripts/slice_audio.py scripts/sync_labeled_images.py scripts/train_baseline.py scripts/train_model.py scripts/verify_model.py scripts/verify_spectrograms.py
/c/Users/ghimi/.local/bin/uv.exe lock --check
git diff --check
```

Scan changed text files for the em dash with Python, reading bytes.

Expected: everything passes.

- [ ] **Step 4: Commit**

```bash
git add tests/test_repository_reproducibility.py docs README.md
git commit -m "docs: document ResNet training and verification"
```

---

### Task 11: Real training run and recorded results

**Files:**
- Modify: `roadmap.md`, `AGENTS.md`, `docs/outline.md`, `README.md` (status and results only)

- [ ] **Step 1: Rebuild the manifest and train**

```bash
.venv/Scripts/python.exe scripts/build_manifest.py
.venv/Scripts/python.exe scripts/train_model.py
```

Run the training in the background and post short status lines while it runs. Expected: a run folder under `models/` and a printed test summary.

- [ ] **Step 2: Verify on the GPU and the CPU**

```bash
.venv/Scripts/python.exe scripts/verify_model.py models/<run folder>
.venv/Scripts/python.exe scripts/verify_model.py models/<run folder> --device cpu
```

Expected: `PASSED` twice.

- [ ] **Step 3: Read the report and judge it**

Read `report.md`. Check that the ResNet beats the baseline in every rotation, that the cross-validation spread is small, and whether errors cluster by hour or in faint clips. If the ResNet does not beat the baseline, stop and report instead of recording success.

- [ ] **Step 4: Record the results**

- `roadmap.md`: update the Current checkpoint counts; Phase 3 status becomes "In progress" with the October 14 counts; Phase 4 gets a "Verified acceptance" block with the run name, test metrics, threshold, cross-validation spread, false detections per hour, faint recall, and both verification results, and its status becomes "Complete" only if every acceptance criterion holds; Phase 5 becomes "Status: Next" if Phase 4 is complete.
- `AGENTS.md`, `docs/outline.md`, and `README.md`: replace the statements that Phase 3 is next and that only a baseline exists.

- [ ] **Step 5: Final checks and commit**

Run the full suite, `git diff --check`, the em dash scan, and `git status --short`, then:

```bash
git add roadmap.md AGENTS.md docs/outline.md README.md
git commit -m "docs: record the first ResNet training results"
```
