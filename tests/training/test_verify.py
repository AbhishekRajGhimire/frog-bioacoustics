from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from frog_classifier.training.artifact import PREDICTIONS_FILE, read_csv
from frog_classifier.training.run import train_and_save
from frog_classifier.training.verify import PROBABILITY_TOLERANCE, verify_run
from tests.training.helpers import build_synthetic_repo


CPU = torch.device("cpu")


class VerifyTests(unittest.TestCase):
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

    def test_a_fresh_run_reproduces_its_predictions(self) -> None:
        result = verify_run(self.run_dir, self.repo, device=CPU)

        self.assertTrue(result.passed, result.problems)
        self.assertEqual(result.checked, 20)
        self.assertLessEqual(result.largest_difference, PROBABILITY_TOLERANCE)

    def test_a_changed_image_is_reported_as_changed(self) -> None:
        row = read_csv(self.run_dir / PREDICTIONS_FILE)[0]
        Image.fromarray(np.full((128, 216), 7, dtype=np.uint8)).save(self.repo / row["image_path"])

        result = verify_run(self.run_dir, self.repo, device=CPU)

        self.assertFalse(result.passed)
        self.assertEqual(result.problems, (f"{row['example_id']}: image changed since training",))
        self.assertEqual(result.checked, 19)

    def test_a_relabeled_clip_is_reported_as_missing(self) -> None:
        row = read_csv(self.run_dir / PREDICTIONS_FILE)[0]
        (self.repo / row["image_path"]).unlink()

        result = verify_run(self.run_dir, self.repo, device=CPU)

        self.assertEqual(result.problems, (f"{row['example_id']}: image missing at {row['image_path']}",))

    def test_a_tampered_run_file_fails_its_checksum(self) -> None:
        predictions = self.run_dir / PREDICTIONS_FILE
        predictions.write_bytes(predictions.read_bytes().replace(b"0.", b"1.", 1))

        result = verify_run(self.run_dir, self.repo, device=CPU)

        self.assertEqual(result.problems, ("predictions.csv: checksum differs from run.json",))
        self.assertEqual(result.checked, 0)


if __name__ == "__main__":
    unittest.main()
