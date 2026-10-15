from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frog_classifier.packaging import selfcheck
from frog_classifier.packaging.selfcheck import SelfCheckError
from frog_classifier.training.artifact import MODEL_FILE
from tests.inference.helpers import build_saved_run


APP = Path(__file__).resolve().parents[2] / "classifier_app" / "app.py"


class SelfCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_an_import_location_outside_the_package_fails(self) -> None:
        package = self.root / "Frog Call Scanner"
        python = package / "app" / "python"
        python.mkdir(parents=True)

        selfcheck.check_isolated(package, [str(python / "Lib"), str(python)])
        with self.assertRaisesRegex(SelfCheckError, "outside the package"):
            selfcheck.check_isolated(package, [str(python), str(self.root / "elsewhere")])

    def test_a_graphics_card_pytorch_fails(self) -> None:
        with patch("torch.version.cuda", "13.0"):
            with self.assertRaisesRegex(SelfCheckError, "graphics-card build"):
                selfcheck.check_cpu_pytorch()
        with patch("torch.version.cuda", None):
            selfcheck.check_cpu_pytorch()

    def test_the_app_modules_leave_pytorch_unloaded(self) -> None:
        selfcheck.check_app_without_pytorch()

    def test_a_changed_model_file_fails(self) -> None:
        run_dir = build_saved_run(self.root / "models")
        self.assertEqual(selfcheck.check_model(self.root / "models"), run_dir)

        (run_dir / MODEL_FILE).write_bytes(b"changed")

        with self.assertRaisesRegex(SelfCheckError, "cannot be used"):
            selfcheck.check_model(self.root / "models")

    def test_a_ten_second_recording_is_scanned_into_two_windows(self) -> None:
        selfcheck.check_scan(build_saved_run(self.root / "models"))

    def test_the_app_page_runs_and_names_the_packaged_model(self) -> None:
        build_saved_run(self.root / "models")

        selfcheck.check_app_renders(APP, self.root / "models")
