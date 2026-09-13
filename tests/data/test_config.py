from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from frog_classifier.data.config import ConfigError, load_preprocessing_config


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


SCHEMA_ONE_CONFIG = b'''schema_version = 1

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
            self.load(SCHEMA_ONE_CONFIG)

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


if __name__ == "__main__":
    unittest.main()
