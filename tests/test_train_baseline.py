from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from frog_classifier.data.manifest import ManifestRow, serialize_manifest
from frog_classifier.data.validation import ManifestValidationError
from tests.data.helpers import load_test_config


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_baseline.py"


class TrainBaselineManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.repo_root = Path(temporary_directory.name)
        self.config = load_test_config(self.repo_root)
        self.rows = self._valid_rows()
        for row in self.rows:
            self._write_image(row.image_path, row.label)

    def test_trains_on_a_validated_manifest(self) -> None:
        manifest_path = self._write_manifest(self.rows)
        module = self._load_module()
        fitted = []

        def fit(classifier, features, labels):
            fitted.append((features, labels))
            classifier.classes_ = np.array([0, 1])
            classifier.coef_ = np.ones((1, features.shape[1]), dtype=np.float64)
            classifier.intercept_ = np.array([-features.shape[1] / 2])
            classifier.n_features_in_ = features.shape[1]
            return classifier

        with patch.object(module.LogisticRegression, "fit", autospec=True, side_effect=fit):
            self.assertEqual(self._run_main(module, manifest_path), 0)

        self.assertEqual(len(fitted), 1)
        self.assertEqual(fitted[0][0].shape[0], 6)
        self.assertEqual(set(fitted[0][1]), {0, 1})

    def test_rejects_a_manifest_with_a_missing_image_without_fitting(self) -> None:
        invalid_rows = list(self.rows)
        invalid_rows[0] = replace(invalid_rows[0], image_path="images/missing.png")

        self._assert_rejected_without_fit(tuple(invalid_rows))

    def test_rejects_a_manifest_without_a_validation_partition_without_fitting(self) -> None:
        invalid_rows = tuple(
            replace(row, split="train") if row.split == "val" else row
            for row in self.rows
        )

        self._assert_rejected_without_fit(invalid_rows)

    def test_rejects_a_manifest_with_a_single_class_test_partition_without_fitting(self) -> None:
        invalid_rows = list(self.rows)
        test_index = next(
            index
            for index, row in enumerate(invalid_rows)
            if row.split == "test" and row.label == 0
        )
        invalid_rows[test_index] = replace(
            invalid_rows[test_index],
            label=1,
            label_name="litoria_aurea",
        )

        self._assert_rejected_without_fit(tuple(invalid_rows))

    def test_rejects_a_manifest_with_recording_split_leakage_without_fitting(self) -> None:
        invalid_rows = list(self.rows)
        train_index = next(
            index for index, row in enumerate(invalid_rows) if row.split == "train"
        )
        invalid_rows[train_index] = replace(
            invalid_rows[train_index],
            recording_id="litoria_aurea_0",
            start_s=5,
        )

        self._assert_rejected_without_fit(tuple(invalid_rows))

    def test_rejects_a_manifest_with_tampered_identity_without_fitting(self) -> None:
        invalid_rows = list(self.rows)
        invalid_rows[0] = replace(invalid_rows[0], example_id="tampered_start0s")

        self._assert_rejected_without_fit(tuple(invalid_rows))

    def _assert_rejected_without_fit(self, rows: tuple[ManifestRow, ...]) -> None:
        manifest_path = self._write_manifest(rows)
        module = self._load_module()

        with patch.object(module.LogisticRegression, "fit", autospec=True) as fit:
            with self.assertRaises(ManifestValidationError):
                self._run_main(module, manifest_path)

        fit.assert_not_called()

    def _load_module(self):
        module_name = f"train_baseline_test_{id(self)}"
        specification = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)
        specification.loader.exec_module(module)
        module.__file__ = str(self.repo_root / "scripts" / "train_baseline.py")
        return module

    def _run_main(self, module, manifest_path: Path) -> int:
        with patch.object(sys, "argv", ["train_baseline.py", "--manifest", str(manifest_path), "--size", "2"]):
            return module.main()

    def _write_manifest(self, rows: tuple[ManifestRow, ...]) -> Path:
        manifest_path = self.repo_root / "manifest.csv"
        manifest_path.write_bytes(serialize_manifest(rows))
        return manifest_path

    def _write_image(self, image_path: str, label: int) -> None:
        path = self.repo_root / image_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (2, 2), color=255 * label).save(path)

    def _valid_rows(self) -> tuple[ManifestRow, ...]:
        rows = []
        for fold in range(5):
            split = "test" if fold == 0 else "val" if fold == 1 else "train"
            for label_name, label in (("litoria_aurea", 1), ("non_target", 0)):
                recording_id = f"{label_name}_{fold}"
                example_id = f"{recording_id}_start0s"
                rows.append(ManifestRow(
                    manifest_version=1,
                    example_id=example_id,
                    image_path=f"labeled/{label_name}/{example_id}.png",
                    label=label,
                    label_name=label_name,
                    recording_id=recording_id,
                    start_s=0,
                    fold=fold,
                    split=split,
                    preprocessing_config_sha256=self.config.sha256,
                ))
        return tuple(rows)


if __name__ == "__main__":
    unittest.main()
