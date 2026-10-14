from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frog_classifier.inference.export import (
    DETECTIONS_CSV,
    RECORDINGS_CSV,
    ExportError,
    export_csv,
)
from frog_classifier.inference.store import DATABASE_FILE, DetectionStore
from frog_classifier.training.artifact import read_csv
from tests.inference.helpers import SCAN_IDENTITY, recording_row


FAILED_PATH = "site/été 2.wav"


class ExportTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.store = DetectionStore.open(self.output / DATABASE_FILE, SCAN_IDENTITY, now="2026-10-15T00:00:00Z")
        self.addCleanup(self.store.close)
        self.store.save(recording_row("site/20220123_010000.WAV", windows=3), (0, 5, 10), (0.2, 0.5, 0.9))
        self.store.save(
            recording_row(
                FAILED_PATH, windows=0, status="failed", message="LibsndfileError: bad header",
                started_at=None, utc_offset=None, duration_s=None, audio_sha256=None,
            ),
            (),
            (),
        )

    def test_detections_are_the_windows_at_or_above_the_threshold_with_their_provenance(self) -> None:
        export_csv(self.store, self.output)

        rows = read_csv(self.output / DETECTIONS_CSV)
        self.assertEqual([row["window_start_s"] for row in rows], ["5", "10"])
        self.assertEqual(rows[0], {
            "recording_path": "site/20220123_010000.WAV",
            "recording_id": "20220123_010000",
            "window_start_s": "5",
            "window_end_s": "10",
            "window_start_time": "2022-01-23T01:00:05",
            "utc_offset": "+11:00",
            "probability": "0.50000000",
            "threshold": "0.50000000",
            "model_run": SCAN_IDENTITY.model_run,
            "model_sha256": SCAN_IDENTITY.model_sha256,
            "preprocessing_sha256": SCAN_IDENTITY.preprocessing_sha256,
        })

    def test_every_recording_is_listed_with_its_status_and_detection_count(self) -> None:
        export_csv(self.store, self.output)

        rows = read_csv(self.output / RECORDINGS_CSV)
        self.assertEqual(
            [(row["recording_path"], row["status"], row["windows"], row["detections"]) for row in rows],
            [("site/20220123_010000.WAV", "ok", "3", "2"), (FAILED_PATH, "failed", "0", "0")],
        )
        self.assertEqual(rows[0]["started_at"], "2022-01-23T01:00:00")
        self.assertEqual(rows[0]["duration_s"], "10.000")
        self.assertEqual(rows[1]["started_at"], "")
        self.assertEqual(rows[1]["message"], "LibsndfileError: bad header")

    def test_a_csv_that_cannot_be_replaced_is_explained_and_leaves_no_partial_file(self) -> None:
        with patch("frog_classifier.inference.export.os.replace", side_effect=PermissionError(13, "Permission denied")):
            with self.assertRaisesRegex(ExportError, "detections.csv could not be replaced.*run the same command again"):
                export_csv(self.store, self.output)

        self.assertEqual(sorted(path.name for path in self.output.iterdir()), [DATABASE_FILE])


if __name__ == "__main__":
    unittest.main()
