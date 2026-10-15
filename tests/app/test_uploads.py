from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from frog_classifier.app.uploads import SavedUploads, UploadError, save_uploads


class SaveUploadsTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name) / "uploads" / "card"

    def test_recordings_are_saved_under_their_own_names(self) -> None:
        result = save_uploads([("C:\\fakepath\\20220123_010000.WAV", b"one"), ("site/b.mp3", b"two")], self.folder)

        self.assertEqual(result, SavedUploads(saved=("20220123_010000.WAV", "b.mp3"), skipped=()))
        self.assertEqual((self.folder / "20220123_010000.WAV").read_bytes(), b"one")
        self.assertEqual((self.folder / "b.mp3").read_bytes(), b"two")

    def test_an_identical_recording_is_skipped(self) -> None:
        save_uploads([("a.wav", b"one")], self.folder)

        result = save_uploads([("a.wav", b"one"), ("b.wav", b"two")], self.folder)

        self.assertEqual(result, SavedUploads(saved=("b.wav",), skipped=("a.wav",)))

    def test_a_different_recording_under_a_saved_name_is_refused_and_nothing_is_saved(self) -> None:
        save_uploads([("a.wav", b"one")], self.folder)

        with self.assertRaisesRegex(UploadError, "already holds different recordings named a.wav"):
            save_uploads([("b.wav", b"two"), ("a.wav", b"changed")], self.folder)

        self.assertEqual((self.folder / "a.wav").read_bytes(), b"one")
        self.assertFalse((self.folder / "b.wav").exists())

    def test_files_that_are_not_recordings_are_refused(self) -> None:
        with self.assertRaisesRegex(UploadError, "only .wav and .mp3 recordings can be scanned, not notes.txt"):
            save_uploads([("notes.txt", b"x")], self.folder)

    def test_one_name_uploaded_twice_is_refused(self) -> None:
        with self.assertRaisesRegex(UploadError, "a.wav is in this upload more than once"):
            save_uploads([("x/a.wav", b"one"), ("y/A.WAV", b"two")], self.folder)


if __name__ == "__main__":
    unittest.main()
