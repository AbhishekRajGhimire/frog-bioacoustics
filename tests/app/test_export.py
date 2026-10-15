from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from frog_classifier.analytics.analyze import OUTPUT_FILES
from frog_classifier.app.export import results_zip
from frog_classifier.inference.export import DETECTIONS_CSV, RECORDINGS_CSV, export_csv
from frog_classifier.inference.store import DATABASE_FILE, DetectionStore
from tests.analytics.helpers import build_record, recording


class ResultsZipTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.record = build_record(Path(directory.name) / "card-run", [
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.95, 0.6, 0.1)),
        ])
        with DetectionStore.open_read_only(self.record / DATABASE_FILE) as store:
            export_csv(store, self.record)

    def test_the_zip_holds_the_analysis_and_both_spreadsheets_in_one_folder(self) -> None:
        with zipfile.ZipFile(io.BytesIO(results_zip(self.record))) as archive:
            names = set(archive.namelist())

        self.assertEqual(names, {f"card-run/{name}" for name in (*OUTPUT_FILES, DETECTIONS_CSV, RECORDINGS_CSV)})

    def test_the_analysis_uses_the_chosen_threshold_and_offset(self) -> None:
        with zipfile.ZipFile(io.BytesIO(results_zip(self.record, threshold=0.9, utc_offset="+10:00"))) as archive:
            report = archive.read("card-run/report.md").decode("utf-8")

        self.assertIn("(1 of 3 windows of 5 seconds) at threshold 0.900", report)
        self.assertIn("| Time basis | UTC+10:00 |", report)


if __name__ == "__main__":
    unittest.main()
