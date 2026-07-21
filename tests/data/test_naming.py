from pathlib import Path
import unittest

from frog_classifier.data.naming import ExampleNameError, parse_example_filename


class ParseExampleFilenameTests(unittest.TestCase):
    def test_parses_canonical_filename(self) -> None:
        simple = parse_example_filename(
            Path("20220308_050000_start30s.png"),
            chunk_seconds=5,
        )
        prefixed = parse_example_filename(
            Path("2MM03935_20251130_180000_start230s.png"),
            chunk_seconds=5,
        )

        self.assertEqual((simple.recording_id, simple.start_s), ("20220308_050000", 30))
        self.assertEqual(prefixed.example_id, "2MM03935_20251130_180000_start230s")

    def test_rejects_noncanonical_filenames(self) -> None:
        for filename in (
            "recording.png",
            "_start5s.png",
            "recording_start-5s.png",
            "recording_start2.5s.png",
            "recording_start6s.png",
            "recording_start5s.jpg",
        ):
            with self.subTest(filename=filename):
                with self.assertRaises(ExampleNameError):
                    parse_example_filename(Path(filename), chunk_seconds=5)

    def test_rejects_nonpositive_chunk_seconds(self) -> None:
        with self.assertRaisesRegex(ValueError, "chunk_seconds must be positive"):
            parse_example_filename(Path("recording_start0s.png"), chunk_seconds=0)
