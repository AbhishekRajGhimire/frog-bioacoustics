from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from frog_classifier.app.scans import ScanStatus, launch_scan_process, read_status, request_stop, write_status
from frog_classifier.inference.store import DATABASE_FILE, DetectionStore
from tests.inference.helpers import AUDIOMOTH_COMMENT, build_saved_run, write_recording
from tests.preprocessing.helpers import noise


TIMEOUT_S = 300


class ScanJobTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.run_dir = build_saved_run(self.root / "models")
        self.card = self.root / "card"
        for seed, name in enumerate(("20220123_010000.WAV", "20220123_020000.WAV", "20220124_010000.WAV")):
            write_recording(self.card / name, noise(6, sample_rate_hz=48000, seed=seed), comment=AUDIOMOTH_COMMENT)
        self.record = self.root / "results" / "detections" / "card-run"
        self.record.mkdir(parents=True)

    def _run(self, input_root: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "frog_classifier.app.scan_job", str(self.run_dir), str(input_root), str(self.record),
             "--workers", "1"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )

    def _stored(self) -> int:
        if not (self.record / DATABASE_FILE).is_file():
            return 0
        with DetectionStore.open_read_only(self.record / DATABASE_FILE) as store:
            return len(store.recording_rows())

    def test_a_scan_stores_every_recording_and_reports_finished(self) -> None:
        result = self._run(self.card)

        self.assertEqual(result.returncode, 0, result.stderr)
        status = read_status(self.record)
        self.assertEqual(
            (status.state, status.found, status.processed, status.failed, status.results_ready),
            ("finished", 3, 3, 0, True),
        )
        self.assertIn(status.device, ("cpu", "cuda"))
        self.assertEqual(self._stored(), 3)

    def test_a_stop_request_keeps_only_whole_recordings(self) -> None:
        process = launch_scan_process(self.run_dir, self.card, self.record, workers=1)
        # Requested before the process has loaded PyTorch, so it never reaches the last recording.
        request_stop(self.record)

        self.assertEqual(process.wait(timeout=TIMEOUT_S), 130)
        status = read_status(self.record)
        self.assertEqual((status.state, status.results_ready), ("stopped", False))
        self.assertLess(self._stored(), 3)

    def test_a_scan_that_cannot_start_says_why_and_keeps_earlier_results(self) -> None:
        write_status(self.record, ScanStatus("starting", "2026-10-15T00:00:00Z", "2026-10-15T00:00:00Z", True))

        result = self._run(self.root / "missing")

        self.assertEqual(result.returncode, 1)
        status = read_status(self.record)
        self.assertEqual((status.state, status.results_ready), ("failed", True))
        self.assertIn("input folder not found", status.message)


if __name__ == "__main__":
    unittest.main()
