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
