from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from frog_classifier.training.artifact import check_files, read_csv, read_run
from frog_classifier.training.run import train_and_save
from tests.training.helpers import build_synthetic_repo


CPU = torch.device("cpu")
FIXED_TIME = datetime(2026, 10, 14, 1, 2, 3, tzinfo=timezone.utc)


def _quiet(_: str) -> None:
    pass


class CrossValidatedRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.paths = build_synthetic_repo(Path(cls.directory.name))
        cls.messages: list[str] = []
        cls.run_dir = train_and_save(cls.paths, device=CPU, progress=cls.messages.append, clock=lambda: FIXED_TIME)
        cls.metadata = read_run(cls.run_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_writes_every_file_under_the_final_name_only(self) -> None:
        self.assertEqual(self.run_dir.name, "resnet18-20261014T010203Z")
        self.assertEqual(
            sorted(path.name for path in self.run_dir.iterdir()),
            sorted([
                "cross_validation.csv", "history.csv", "manifest.csv", "model.pt",
                "out_of_fold_predictions.csv", "precision_recall.png", "predictions.csv",
                "preprocessing.toml", "report.md", "run.json", "training.toml",
            ]),
        )
        self.assertEqual([path.name for path in self.paths.models_root.iterdir()], [self.run_dir.name])
        self.assertEqual(check_files(self.run_dir, self.metadata), [])

    def test_copies_inputs_and_records_their_checksums(self) -> None:
        self.assertEqual((self.run_dir / "manifest.csv").read_bytes(), self.paths.manifest.read_bytes())
        self.assertEqual(
            self.metadata["checksums"]["manifest"],
            hashlib.sha256(self.paths.manifest.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            (self.run_dir / "training.toml").read_bytes(), self.paths.training_config.read_bytes()
        )

    def test_predictions_cover_every_example_with_its_image_checksum(self) -> None:
        rows = read_csv(self.run_dir / "predictions.csv")

        self.assertEqual(len(rows), 20)
        for row in rows:
            image = self.paths.repo_root / row["image_path"]
            self.assertEqual(row["image_sha256"], hashlib.sha256(image.read_bytes()).hexdigest())
            self.assertTrue(0.0 <= float(row["probability"]) <= 1.0)
            self.assertTrue(0.0 <= float(row["baseline_probability"]) <= 1.0)
        self.assertEqual(sorted(row["faint"] for row in rows if row["label"] == "1"), ["0"] * 9 + ["1"])
        self.assertTrue(all(row["faint"] == "" for row in rows if row["label"] == "0"))

    def test_every_fold_is_tested_once_and_every_example_gets_one_out_of_fold_probability(self) -> None:
        rotations = self.metadata["cross_validation"]["rotations"]
        self.assertEqual(sorted(row["test_fold"] for row in rotations), [0, 1, 2, 3, 4])
        self.assertEqual(rotations[0]["test_fold"], self.metadata["data"]["test_fold"])
        out_of_fold = read_csv(self.run_dir / "out_of_fold_predictions.csv")
        self.assertEqual(len(out_of_fold), 20)
        self.assertEqual(len({row["example_id"] for row in out_of_fold}), 20)
        self.assertEqual(len(read_csv(self.run_dir / "cross_validation.csv")), 10)

    def test_metadata_records_the_run(self) -> None:
        self.assertEqual(self.metadata["data"]["faint_tagged_frog_clips"], 1)
        self.assertEqual(self.metadata["metrics"]["faint_recall"]["test"]["faint"]["clips"], 1)
        self.assertEqual(self.metadata["training"]["device"], "cpu")
        self.assertIsNone(self.metadata["model"]["pretrained_weights"])
        self.assertEqual(self.metadata["model"]["input_shape"], [1, 128, 216])
        self.assertIsNone(self.metadata["source"]["git_commit"])
        self.assertEqual(self.metadata["data"]["splits"]["test"]["recordings"], 2)
        self.assertTrue(0.0 <= self.metadata["threshold"]["value"] <= 1.0)
        self.assertIn("Cross-validation", (self.run_dir / "report.md").read_text(encoding="utf-8"))
        self.assertTrue(any(message.startswith("epoch") for message in self.messages))


class SingleRunTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_skipping_cross_validation_trains_once_and_writes_no_rotation_files(self) -> None:
        paths = build_synthetic_repo(self.root)

        run_dir = train_and_save(paths, cross_validate=False, device=CPU, progress=_quiet)

        self.assertIsNone(read_run(run_dir)["cross_validation"])
        self.assertFalse((run_dir / "cross_validation.csv").exists())
        self.assertFalse((run_dir / "out_of_fold_predictions.csv").exists())

    def test_rotation_zero_follows_a_manifest_with_other_folds(self) -> None:
        paths = build_synthetic_repo(self.root, test_fold=2, val_fold=3)

        metadata = read_run(train_and_save(paths, cross_validate=False, device=CPU, progress=_quiet))

        self.assertEqual((metadata["data"]["test_fold"], metadata["data"]["validation_fold"]), (2, 3))

    def test_trains_without_a_decision_log(self) -> None:
        paths = build_synthetic_repo(self.root, faint_log=False)

        metadata = read_run(train_and_save(paths, cross_validate=False, device=CPU, progress=_quiet))

        self.assertEqual(metadata["data"]["faint_tagged_frog_clips"], 0)

    def test_records_inputs_and_source_as_they_were_when_training_started(self) -> None:
        paths = build_synthetic_repo(self.root)
        git = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "core.autocrlf=false"]
        subprocess.run([*git, "init", "-q"], cwd=self.root, check=True)
        subprocess.run([*git, "add", "-A"], cwd=self.root, check=True)
        subprocess.run([*git, "commit", "-q", "-m", "synthetic"], cwd=self.root, check=True)
        original_manifest = paths.manifest.read_bytes()
        original_training = paths.training_config.read_bytes()

        def edit_inputs_mid_run(message: str) -> None:
            # A manifest rebuild or a config edit while the model trains must not leak into the record.
            if message.startswith("epoch") and paths.manifest.read_bytes() == original_manifest:
                paths.manifest.write_bytes(original_manifest + b"\n")
                paths.training_config.write_bytes(original_training + b"\n# edited\n")

        run_dir = train_and_save(paths, cross_validate=False, device=CPU, progress=edit_inputs_mid_run)

        metadata = read_run(run_dir)
        self.assertEqual((run_dir / "manifest.csv").read_bytes(), original_manifest)
        self.assertEqual((run_dir / "training.toml").read_bytes(), original_training)
        self.assertEqual(metadata["checksums"]["manifest"], hashlib.sha256(original_manifest).hexdigest())
        self.assertFalse(metadata["source"]["git_worktree_dirty"])
        self.assertEqual(len(metadata["source"]["git_commit"]), 40)

    def test_refuses_an_existing_run_folder_before_training(self) -> None:
        paths = build_synthetic_repo(self.root)
        (paths.models_root / "resnet18-20261014T010203Z").mkdir(parents=True)
        messages: list[str] = []

        with self.assertRaises(FileExistsError):
            train_and_save(paths, device=CPU, progress=messages.append, clock=lambda: FIXED_TIME)

        self.assertFalse(any(message.startswith("epoch") for message in messages))

    def test_a_failed_run_leaves_no_folder_under_a_final_name(self) -> None:
        paths = build_synthetic_repo(self.root)
        first_image = sorted((paths.labeled_root / "non_target").glob("*.png"))[0]
        Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(first_image)

        with self.assertRaisesRegex(ValueError, first_image.name):
            train_and_save(paths, device=CPU, progress=_quiet, clock=lambda: FIXED_TIME)

        self.assertEqual(
            [path.name for path in paths.models_root.iterdir()],
            [".resnet18-20261014T010203Z.partial"],
        )


if __name__ == "__main__":
    unittest.main()
