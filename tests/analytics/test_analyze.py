from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from frog_classifier.analytics.analyze import (
    DATE_TABLE,
    HOUR_TABLE,
    OUTPUT_FILES,
    WINDOWS_TABLE,
    analyze_record,
    default_analysis_dir,
)
from frog_classifier.inference.store import StoreError
from frog_classifier.training.artifact import read_csv
from tests.analytics.helpers import build_record, recording


class AnalyzeRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.record = build_record(self.root / "record", [
            recording("a/20220120_010000.WAV", "2022-01-20T01:00:00", (0.9, 0.1, 0.6)),
            recording("a/20220122_235955.WAV", "2022-01-22T23:59:55", (0.95, 0.2, 0.7)),
            recording("a/20220123_020000.WAV", "2022-01-23T02:00:00", (0.1, 0.1)),
        ])
        self.output = self.root / "analysis"

    def test_every_output_is_written(self) -> None:
        analyze_record(self.record, self.output)

        self.assertEqual(sorted(path.name for path in self.output.iterdir()), sorted(OUTPUT_FILES))

    def test_every_hour_and_date_value_is_reproduced_from_the_windows_table(self) -> None:
        analyze_record(self.record, self.output)

        windows = read_csv(self.output / WINDOWS_TABLE)
        self.assertEqual(len(windows), 8)
        for table, key in ((HOUR_TABLE, "hour"), (DATE_TABLE, "date")):
            for row in read_csv(self.output / table):
                with self.subTest(table=table, bin=row[key]):
                    matching = [window for window in windows if window[key] == row[key]]
                    detected = sum(int(window["detected"]) for window in matching)
                    self.assertEqual(int(row["recorded_windows"]), len(matching))
                    self.assertEqual(int(row["detected_windows"]), detected)
                    self.assertEqual(int(row["recordings"]), len({window["recording_path"] for window in matching}))
                    expected_share = f"{detected / len(matching):.6f}" if matching else ""
                    self.assertEqual(row["calling_share"], expected_share)

    def test_the_detection_record_is_left_unchanged(self) -> None:
        before = {path.name: path.read_bytes() for path in self.record.iterdir()}

        analyze_record(self.record, self.output)

        self.assertEqual({path.name: path.read_bytes() for path in self.record.iterdir()}, before)

    def test_a_folder_without_a_record_is_refused(self) -> None:
        with self.assertRaisesRegex(StoreError, "run the detection command first"):
            analyze_record(self.root / "empty", self.output)

        self.assertFalse(self.output.exists())

    def test_the_default_output_folder_is_named_after_the_record(self) -> None:
        self.assertEqual(default_analysis_dir(self.root / "results", self.record), self.root / "results" / "record")

    def test_analysis_does_not_load_pytorch(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import sys, frog_classifier.analytics.analyze; print('torch' in sys.modules)"],
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
