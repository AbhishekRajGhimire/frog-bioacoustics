from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from frog_classifier.data.config import ConfigError, load_preprocessing_config


VALID_CONFIG = b'''schema_version = 1

[audio]
sample_rate_hz = 22050
mono = true
chunk_seconds = 5
overlap_seconds = 0
drop_incomplete_final_chunk = true

[spectrogram]
kind = "mel"
n_mels = 128
fmin_hz = 0
fmax_hz = 8000
power = 2.0
n_fft = 2048
hop_length = 512

[rendering]
format = "png"
figure_width_inches = 3.2
figure_height_inches = 3.2
dpi = 150
axis_visible = false
interpolation = "nearest"

[classes]
litoria_aurea = 1
non_target = 0
'''


class LoadPreprocessingConfigTests(unittest.TestCase):
    def load(self, content: bytes):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "preprocessing.toml"
            path.write_bytes(content)
            return load_preprocessing_config(path), path.resolve()

    def test_loads_every_tracked_preprocessing_value(self) -> None:
        config, source_path = self.load(VALID_CONFIG)

        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.audio.sample_rate_hz, 22050)
        self.assertTrue(config.audio.mono)
        self.assertEqual(config.audio.chunk_seconds, 5)
        self.assertEqual(config.audio.overlap_seconds, 0)
        self.assertTrue(config.audio.drop_incomplete_final_chunk)
        self.assertEqual(config.spectrogram.kind, "mel")
        self.assertEqual(config.spectrogram.n_mels, 128)
        self.assertEqual(config.spectrogram.fmin_hz, 0)
        self.assertEqual(config.spectrogram.fmax_hz, 8000)
        self.assertEqual(config.spectrogram.power, 2.0)
        self.assertEqual(config.spectrogram.n_fft, 2048)
        self.assertEqual(config.spectrogram.hop_length, 512)
        self.assertEqual(config.rendering.format, "png")
        self.assertEqual(config.rendering.figure_width_inches, 3.2)
        self.assertEqual(config.rendering.figure_height_inches, 3.2)
        self.assertEqual(config.rendering.dpi, 150)
        self.assertFalse(config.rendering.axis_visible)
        self.assertEqual(config.rendering.interpolation, "nearest")
        self.assertEqual(dict(config.classes), {"litoria_aurea": 1, "non_target": 0})
        self.assertEqual(config.source_path, source_path)
        self.assertEqual(config.sha256, hashlib.sha256(VALID_CONFIG).hexdigest())

    def test_rejects_noncanonical_schema_version(self) -> None:
        with self.assertRaisesRegex(ConfigError, "schema_version"):
            self.load(VALID_CONFIG.replace(b"schema_version = 1", b"schema_version = 2"))

    def test_rejects_missing_rendering_section(self) -> None:
        invalid_config = VALID_CONFIG.replace(
            b'''[rendering]
format = "png"
figure_width_inches = 3.2
figure_height_inches = 3.2
dpi = 150
axis_visible = false
interpolation = "nearest"

''',
            b"",
        )

        with self.assertRaisesRegex(ConfigError, "rendering"):
            self.load(invalid_config)

    def test_rejects_overlap_equal_to_chunk_duration(self) -> None:
        with self.assertRaisesRegex(ConfigError, "overlap"):
            self.load(VALID_CONFIG.replace(b"overlap_seconds = 0", b"overlap_seconds = 5"))

    def test_rejects_frequency_above_nyquist(self) -> None:
        with self.assertRaisesRegex(ConfigError, "frequency"):
            self.load(VALID_CONFIG.replace(b"fmax_hz = 8000", b"fmax_hz = 12000"))

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

    def test_rejects_nonfinite_rendering_float_values(self) -> None:
        fields = (
            ("figure_width_inches", "3.2"),
            ("figure_height_inches", "3.2"),
        )
        for field, original in fields:
            for value in ("nan", "inf", "-inf"):
                with self.subTest(field=field, value=value):
                    invalid = VALID_CONFIG.replace(
                        f"{field} = {original}".encode("ascii"),
                        f"{field} = {value}".encode("ascii"),
                    )

                    with self.assertRaisesRegex(ConfigError, field):
                        self.load(invalid)


if __name__ == "__main__":
    unittest.main()
