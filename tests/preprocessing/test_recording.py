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
