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
