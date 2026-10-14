from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from frog_classifier.inference.recordings import (
    NO_START_TIME,
    audiomoth_comment,
    find_recordings,
    recording_time,
)
from tests.inference.helpers import AUDIOMOTH_COMMENT, riff_chunk, write_recording
from tests.preprocessing.helpers import noise


class FindRecordingsTests(unittest.TestCase):
    def test_finds_audio_files_in_every_subfolder_in_sorted_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "b/20220102_000000.WAV",
                "a/x.mp3",
                "a/deeper/y.wav",
                "a/notes.txt",
                "a/CONFIG.TXT",
                "c d/été 1.Wav",
            ):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_bytes(b"")

            self.assertEqual(
                find_recordings(root),
                ["a/deeper/y.wav", "a/x.mp3", "b/20220102_000000.WAV", "c d/été 1.Wav"],
            )


class RecordingTimeTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _recording(self, name: str, comment: str | None = None, *, extra_chunks: bytes = b"") -> Path:
        return write_recording(
            self.root / name,
            noise(1, sample_rate_hz=48000),
            comment=comment,
            extra_chunks=extra_chunks,
        )

    def test_the_file_name_gives_the_start_time(self) -> None:
        time = recording_time(self._recording("20220123_010000.WAV"))

        self.assertEqual(time.started_at, datetime(2022, 1, 23, 1, 0, 0))
        self.assertIsNone(time.utc_offset)
        self.assertEqual(time.notes, ())

    def test_the_audiomoth_header_gives_the_utc_offset(self) -> None:
        time = recording_time(self._recording("20220123_010000.WAV", AUDIOMOTH_COMMENT))

        self.assertEqual(time.started_at, datetime(2022, 1, 23, 1, 0, 0))
        self.assertEqual(time.utc_offset, "+11:00")
        self.assertEqual(time.notes, ())

    def test_offsets_are_signed_hours_and_minutes(self) -> None:
        cases = (("UTC", "+00:00"), ("UTC-3", "-03:00"), ("UTC+9:30", "+09:30"), ("UTC+10", "+10:00"))
        for index, (zone, expected) in enumerate(cases):
            with self.subTest(zone=zone):
                comment = f"Recorded at 01:00:00 23/01/2022 ({zone}) by AudioMoth 248D9B0260370525."
                path = self._recording(f"zone{index}/20220123_010000.WAV", comment)

                self.assertEqual(recording_time(path).utc_offset, expected)

    def test_the_header_gives_the_start_time_when_the_file_name_has_none(self) -> None:
        time = recording_time(self._recording("site-a.wav", AUDIOMOTH_COMMENT))

        self.assertEqual(time.started_at, datetime(2022, 1, 23, 1, 0, 0))
        self.assertEqual(time.notes, ("start time taken from the AudioMoth header",))

    def test_a_disagreeing_header_is_noted_and_the_file_name_wins(self) -> None:
        time = recording_time(self._recording("20220123_020000.WAV", AUDIOMOTH_COMMENT))

        self.assertEqual(time.started_at, datetime(2022, 1, 23, 2, 0, 0))
        self.assertEqual(time.notes, (
            "file name time 2022-01-23T02:00:00 differs from the AudioMoth header time "
            "2022-01-23T01:00:00; using the file name",
        ))

    def test_an_impossible_file_name_date_is_noted(self) -> None:
        time = recording_time(self._recording("20221340_010000.wav"))

        self.assertIsNone(time.started_at)
        self.assertEqual(time.notes, (
            "file name timestamp 20221340_010000 is not a valid date",
            NO_START_TIME,
        ))

    def test_a_recording_without_any_time_says_why(self) -> None:
        time = recording_time(self._recording("site-a.wav"))

        self.assertIsNone(time.started_at)
        self.assertIsNone(time.utc_offset)
        self.assertEqual(time.notes, (NO_START_TIME,))

    def test_the_comment_is_found_after_other_chunks_of_odd_length(self) -> None:
        other_chunks = riff_chunk(b"junk", b"abc") + riff_chunk(b"LIST", b"adtl" + riff_chunk(b"note", b"x"))
        path = self._recording("20220123_010000.WAV", AUDIOMOTH_COMMENT, extra_chunks=other_chunks)

        self.assertEqual(audiomoth_comment(path), AUDIOMOTH_COMMENT)

    def test_a_file_that_is_not_a_wav_has_no_comment(self) -> None:
        path = self.root / "20220123_010000.mp3"
        path.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00")

        self.assertIsNone(audiomoth_comment(path))


if __name__ == "__main__":
    unittest.main()
