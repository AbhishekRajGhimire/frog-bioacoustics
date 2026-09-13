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

    def test_incomplete_window_is_not_exported(self) -> None:
        write_wav(self.raw_root / "site" / "short.wav", tone(1000.0, 3))
        image = self._touch(self.spectrogram_root / "site" / "short_start0s.png")

        self.assertIsNone(find_or_export_chunk(image, raw_root=self.raw_root, chunk_root=self.chunk_root, config=self.config, spectrogram_root=self.spectrogram_root))
        self.assertFalse((self.chunk_root / "site" / "short_start0s.wav").exists())

    def _touch(self, path: Path, content: bytes = b"") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path


if __name__ == "__main__":
    unittest.main()
