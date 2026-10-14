from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from frog_classifier.training.artifact import MODEL_FILE, RUN_METADATA, file_sha256, read_run, write_json
from frog_classifier.training.run import train_and_save
from frog_classifier.training.saved import SavedRunError, load_saved_run
from tests.training.helpers import build_synthetic_repo


CPU = torch.device("cpu")


class LoadSavedRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.source = Path(cls.directory.name) / "source"
        paths = build_synthetic_repo(cls.source)
        cls.run_name = train_and_save(paths, cross_validate=False, device=CPU, progress=lambda _: None).name

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def setUp(self) -> None:
        # Each test gets its own copy so tampering never leaks into another test.
        working = tempfile.TemporaryDirectory()
        self.addCleanup(working.cleanup)
        self.repo = Path(working.name) / "repo"
        shutil.copytree(self.source, self.repo)
        self.run_dir = self.repo / "models" / self.run_name

    def test_loads_the_model_threshold_and_training_preprocessing(self) -> None:
        saved = load_saved_run(self.run_dir)

        metadata = read_run(self.run_dir)
        self.assertEqual(saved.name, self.run_name)
        self.assertEqual(saved.threshold, metadata["threshold"]["value"])
        self.assertEqual(saved.model_sha256, file_sha256(self.run_dir / MODEL_FILE))
        self.assertEqual(saved.preprocessing.sha256, metadata["checksums"]["preprocessing_config"])
        self.assertFalse(saved.model.training)

    def test_a_changed_model_file_is_refused(self) -> None:
        model = self.run_dir / MODEL_FILE
        model.write_bytes(model.read_bytes() + b"\0")

        with self.assertRaisesRegex(SavedRunError, "model.pt: checksum differs from run.json"):
            load_saved_run(self.run_dir)

    def test_an_unknown_architecture_is_refused(self) -> None:
        metadata = read_run(self.run_dir)
        metadata["model"]["architecture"] = "resnet50"
        write_json(self.run_dir / RUN_METADATA, metadata)

        with self.assertRaisesRegex(SavedRunError, "unsupported architecture 'resnet50'"):
            load_saved_run(self.run_dir)

    def test_a_folder_that_is_not_a_run_is_refused(self) -> None:
        with self.assertRaisesRegex(SavedRunError, "is not a saved run"):
            load_saved_run(self.repo / "labeled")


if __name__ == "__main__":
    unittest.main()
