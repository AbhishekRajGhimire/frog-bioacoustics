from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from frog_classifier.app.deployments import (
    DEPLOYMENT_FILE,
    DeploymentError,
    RecordingsPreview,
    clean_name,
    deployment_dir,
    list_deployments,
    preview_recordings,
    start_scan,
)
from frog_classifier.app.models import ModelInfo
from frog_classifier.app.paths import AppPaths
from frog_classifier.app.scans import ScanStatus, request_stop, stop_requested, write_status
from tests.analytics.helpers import build_record, recording
from tests.inference.helpers import SCAN_IDENTITY


MODEL = ModelInfo(Path("models/resnet18-20261015T000000Z"), "resnet18-20261015T000000Z", 0.5)
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


def finished(results_ready: bool = True) -> ScanStatus:
    return ScanStatus("finished", "2026-10-15T12:00:00Z", "2026-10-15T12:01:00Z", results_ready, found=3, processed=3)


class NameTests(unittest.TestCase):
    def test_names_keep_only_characters_safe_in_folder_names(self) -> None:
        for text, expected in (
            ("IWS076 card 3", "IWS076 card 3"),
            ("site/a: north", "site-a- north"),
            ("  card 1.  ", "card 1"),
            ("Pond (east)_2", "Pond (east)_2"),
        ):
            with self.subTest(text=text):
                self.assertEqual(clean_name(text), expected)

    def test_unusable_names_are_refused(self) -> None:
        for text in ("", " . ", "CON", "lpt1.txt", "x" * 81):
            with self.subTest(text=text), self.assertRaises(DeploymentError):
                clean_name(text)

    def test_a_deployment_folder_ends_with_the_model_run(self) -> None:
        self.assertEqual(deployment_dir(Path("results"), "card 3", "resnet18-x"), Path("results") / "card 3-resnet18-x")


class DeploymentTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.paths = AppPaths(self.root)
        self.card = self.root / "card"
        for name in ("20220123_010000.WAV", "b/20220405_020000.wav", "notes.txt", "c/unnamed.mp3"):
            (self.card / name).parent.mkdir(parents=True, exist_ok=True)
            (self.card / name).write_bytes(b"")

    def _start(self, name: str = "card", folder: Path | None = None, now: datetime = NOW):
        with patch("frog_classifier.app.deployments.launch_scan_process") as launch:
            record_dir = start_scan(self.paths, MODEL, folder or self.card, name, now=now)
        return record_dir, launch

    def test_starting_a_scan_records_the_deployment_and_launches_the_scan(self) -> None:
        record_dir, launch = self._start()

        self.assertEqual(record_dir, self.paths.detections / "card-resnet18-20261015T000000Z")
        launch.assert_called_once_with(MODEL.run_dir, self.card.resolve(), record_dir, workers=None)
        [deployment] = list_deployments(self.paths.detections, now=NOW)
        self.assertEqual(
            (deployment.name, deployment.input_root, deployment.model_run, deployment.created_at),
            ("card", self.card.resolve().as_posix(), MODEL.name, "2026-10-15T12:00:00Z"),
        )
        self.assertEqual(deployment.status.state, "starting")
        self.assertTrue(deployment.scanning)
        self.assertFalse(deployment.results_ready)

    def test_continuing_keeps_the_deployment_and_its_complete_results(self) -> None:
        record_dir, _ = self._start()
        write_status(record_dir, finished())
        request_stop(record_dir)
        later = NOW + timedelta(days=1)

        self._start(now=later)

        [deployment] = list_deployments(self.paths.detections, now=later)
        self.assertEqual(deployment.created_at, "2026-10-15T12:00:00Z")
        self.assertEqual((deployment.status.state, deployment.status.results_ready), ("starting", True))
        self.assertFalse(stop_requested(record_dir))

    def test_a_name_used_for_another_folder_is_refused(self) -> None:
        record_dir, _ = self._start()
        write_status(record_dir, finished())
        other = self.root / "other"
        other.mkdir()

        with self.assertRaisesRegex(DeploymentError, "already used for the recordings in .*card; choose another name"):
            self._start(folder=other)

    def test_only_one_scan_runs_at_a_time(self) -> None:
        self._start()
        other = self.root / "other"
        other.mkdir()

        with self.assertRaisesRegex(DeploymentError, "the scan of card is still running"):
            self._start("other", other)

    def test_a_missing_folder_is_refused(self) -> None:
        with self.assertRaisesRegex(DeploymentError, "there is no folder"):
            self._start("missing", self.root / "missing")

    def test_records_made_by_the_command_are_listed_by_their_folder_name(self) -> None:
        build_record(self.paths.detections / f"external-{SCAN_IDENTITY.model_run}", [
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.9,)),
        ])
        self._start()

        deployments = list_deployments(self.paths.detections, now=NOW)

        self.assertEqual([deployment.name for deployment in deployments], ["card", "external"])
        command = deployments[1]
        self.assertEqual((command.status, command.input_root), (None, SCAN_IDENTITY.input_root))
        self.assertTrue(command.results_ready)

    def test_a_scan_that_stopped_reporting_is_interrupted_and_incomplete(self) -> None:
        self._start()

        [deployment] = list_deployments(self.paths.detections, now=NOW + timedelta(minutes=5))

        self.assertEqual(deployment.status.state, "interrupted")
        self.assertFalse(deployment.scanning)
        self.assertFalse(deployment.results_ready)

    def test_a_damaged_deployment_file_is_skipped(self) -> None:
        record_dir, _ = self._start()
        (record_dir / DEPLOYMENT_FILE).write_bytes(b"{not json")

        self.assertEqual(list_deployments(self.paths.detections, now=NOW), [])

    def test_a_folder_preview_counts_recordings_and_their_dates(self) -> None:
        self.assertEqual(
            preview_recordings(self.card),
            RecordingsPreview(count=3, first_day=date(2022, 1, 23), last_day=date(2022, 4, 5), undated=1),
        )

    def test_previewing_a_missing_folder_is_refused(self) -> None:
        with self.assertRaisesRegex(DeploymentError, "there is no folder"):
            preview_recordings(self.root / "missing")

    def test_the_deployment_file_is_plain_json(self) -> None:
        record_dir, _ = self._start()

        self.assertEqual(json.loads((record_dir / DEPLOYMENT_FILE).read_bytes())["name"], "card")


if __name__ == "__main__":
    unittest.main()
