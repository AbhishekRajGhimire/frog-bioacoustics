from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frog_classifier.app.models import ModelError, ModelInfo, newest_model
from frog_classifier.app.paths import ROOT_VARIABLE, app_paths
from frog_classifier.training.artifact import RUN_METADATA


class AppPathsTests(unittest.TestCase):
    def test_the_root_variable_names_the_data_root(self) -> None:
        with patch.dict(os.environ, {ROOT_VARIABLE: "D:/frog-data"}):
            paths = app_paths(Path("ignored"))

        root = Path("D:/frog-data")
        self.assertEqual(
            (paths.models, paths.uploads, paths.detections),
            (root / "models", root / "raw" / "uploads", root / "results" / "detections"),
        )

    def test_without_the_variable_the_default_root_is_used(self) -> None:
        environment = {name: value for name, value in os.environ.items() if name != ROOT_VARIABLE}
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(app_paths(Path("repo")).root, Path("repo"))

    def test_the_app_modules_do_not_load_pytorch(self) -> None:
        modules = ("deployments", "dialog", "export", "listening", "models", "paths", "scan_job", "scans", "uploads")
        imports = ", ".join(f"frog_classifier.app.{name}" for name in modules)
        result = subprocess.run(
            [sys.executable, "-c", f"import sys, {imports}; print('torch' in sys.modules)"],
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertEqual(result.stdout.strip(), "False")


class NewestModelTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _run(self, name: str, threshold: float) -> Path:
        run_dir = self.root / name
        run_dir.mkdir()
        (run_dir / RUN_METADATA).write_bytes(json.dumps({"run_name": name, "threshold": {"value": threshold}}).encode())
        return run_dir

    def test_the_most_recently_trained_run_is_chosen(self) -> None:
        self._run("resnet18-20261014T114432Z", 0.7)
        newest = self._run("resnet34-20261101T000000Z", 0.8)
        self._run("resnet50-20250101T000000Z", 0.9)

        self.assertEqual(newest_model(self.root), ModelInfo(newest, "resnet34-20261101T000000Z", 0.8))

    def test_unfinished_and_unreadable_runs_are_ignored(self) -> None:
        good = self._run("resnet18-20261014T114432Z", 0.7)
        (self.root / ".resnet18-20261201T000000Z.partial").mkdir()
        broken = self.root / "resnet18-20261202T000000Z"
        broken.mkdir()
        (broken / RUN_METADATA).write_bytes(b"{not json")

        self.assertEqual(newest_model(self.root).run_dir, good)

    def test_a_missing_model_is_explained(self) -> None:
        with self.assertRaisesRegex(ModelError, "train one with scripts/train_model.py"):
            newest_model(self.root / "missing")


if __name__ == "__main__":
    unittest.main()
