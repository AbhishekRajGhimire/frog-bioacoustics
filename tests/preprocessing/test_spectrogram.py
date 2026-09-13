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
