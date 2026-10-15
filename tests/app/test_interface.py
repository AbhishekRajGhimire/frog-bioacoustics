from __future__ import annotations

import os
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from frog_classifier.app.paths import ROOT_VARIABLE
from frog_classifier.app.scans import STOP_FILE, ScanStatus, utc_timestamp, write_status
from tests.analytics.helpers import build_record, recording
from tests.inference.helpers import AUDIOMOTH_COMMENT, build_saved_run, write_recording
from tests.preprocessing.helpers import noise


APP = Path(__file__).resolve().parents[2] / "classifier_app" / "app.py"
SCAN_TIMEOUT_S = 300


class InterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        environment = patch.dict(os.environ, {ROOT_VARIABLE: str(self.root)})
        environment.start()
        self.addCleanup(environment.stop)
        self.run_dir = build_saved_run(self.root / "models")
        self.card = self.root / "card"
        for seed, name in enumerate(("20220123_010000.WAV", "20220124_020000.WAV")):
            write_recording(self.card / name, noise(10, sample_rate_hz=48000, seed=seed), comment=AUDIOMOTH_COMMENT)
        self.folder = f"site-a-{self.run_dir.name}"

    def _app(self, view: str | None = None) -> AppTest:
        app = AppTest.from_file(str(APP), default_timeout=60)
        app.run()
        if view is not None:
            app.radio(key="view").set_value(view).run()
        self._assert_no_exception(app)
        return app

    def _assert_no_exception(self, app: AppTest) -> None:
        self.assertEqual(len(app.exception), 0, [exception.value for exception in app.exception])

    def _record(self) -> Path:
        return build_record(self.root / "results" / "detections" / self.folder, [
            recording("20220123_010000.WAV", "2022-01-23T01:00:00", (0.9, 0.2)),
            recording("20220124_020000.WAV", "2022-01-24T02:00:00", (0.97, 0.1)),
            recording("19700101_000000.WAV", "1970-01-01T00:00:00", (0.9,)),
        ], input_root=self.card.as_posix(), model_run=self.run_dir.name)

    def _scanning(self, record: Path) -> None:
        now = utc_timestamp(datetime.now(timezone.utc))
        write_status(record, ScanStatus("running", now, now, False, found=10, processed=4))

    @staticmethod
    def _texts(elements) -> str:
        return "\n".join(element.value for element in elements)

    def test_a_folder_is_previewed_and_suggests_its_own_name(self) -> None:
        app = self._app()

        app.text_input(key="folder").set_value(str(self.card)).run()

        self.assertIn("Found 2 recordings, dated 23 January 2022 to 24 January 2022.", self._texts(app.success))
        self.assertEqual(app.text_input(key="deployment_name").value, "card")
        self.assertFalse(app.button(key="scan").disabled)

    def test_browse_fills_in_the_chosen_folder(self) -> None:
        app = self._app()

        with patch("frog_classifier.app.dialog.choose_folder", return_value=str(self.card)):
            app.button(key="browse").click().run()

        self.assertEqual(app.text_input(key="folder").value, str(self.card))

    def test_the_upload_box_is_offered(self) -> None:
        app = self._app()

        app.radio(key="source").set_value("Upload a few files").run()

        self.assertEqual(len(app.get("file_uploader")), 1)
        self.assertTrue(app.button(key="scan").disabled)

    def test_a_scan_runs_to_the_end_and_opens_its_results(self) -> None:
        app = self._app()
        app.text_input(key="folder").set_value(str(self.card)).run()

        app.button(key="scan").click().run()
        deadline = time.monotonic() + SCAN_TIMEOUT_S
        while "finished" not in self._texts(app.success):
            self._assert_no_exception(app)
            self.assertLess(time.monotonic(), deadline, "the scan did not finish in time")
            time.sleep(1)
            app.run()
        app.button(key="view_results").click().run()

        self.assertEqual(app.radio(key="view").value, "Results")
        self.assertIn("windows of 5 seconds", self._texts(app.markdown))

    def test_a_running_scan_can_be_stopped(self) -> None:
        record = self._record()
        self._scanning(record)
        app = self._app()

        app.button(key="stop").click().run()

        self.assertTrue((record / STOP_FILE).exists())

    def test_results_show_the_summary_warnings_charts_and_download(self) -> None:
        self._record()

        app = self._app("Results")

        self.assertIn(
            "Calls were detected in 50.0% of recorded time (2 of 4 windows of 5 seconds) at threshold 0.500.",
            self._texts(app.markdown),
        )
        self.assertIn("impossible date", self._texts(app.warning))
        self.assertIn("one site", self._texts(app.info))
        self.assertGreaterEqual(len(app.get("imgs")), 3)
        self.assertEqual(len(app.get("download_button")), 1)

    def test_a_new_threshold_changes_the_results(self) -> None:
        self._record()
        app = self._app("Results")

        app.slider(key=f"threshold:{self.folder}").set_value(0.95).run()

        self.assertIn("(1 of 4 windows of 5 seconds) at threshold 0.950.", self._texts(app.markdown))

    def test_the_windows_behind_an_hour_are_listed_and_can_be_heard(self) -> None:
        self._record()

        app = self._app("Results")

        self.assertEqual(app.selectbox(key=f"hour:{self.folder}").value, 1)
        self.assertEqual(len(app.dataframe[0].value), 2)
        self.assertEqual(len(app.get("audio")), 1)

    def test_a_deployment_still_scanning_shows_progress_instead_of_results(self) -> None:
        self._scanning(self._record())

        app = self._app("Results")

        self.assertIn("still being scanned (4 of 10 recordings so far)", self._texts(app.info))
        self.assertEqual(len(app.get("imgs")), 0)


if __name__ == "__main__":
    unittest.main()
