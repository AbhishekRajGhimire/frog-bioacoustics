from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from frog_classifier.data.manifest import (
    LabeledExample,
    ManifestRow,
    build_manifest_rows,
)
from frog_classifier.data.splitting import SplitPlan
from frog_classifier.data.validation import (
    ManifestValidationError,
    validate_manifest_rows,
)


CLASSES = MappingProxyType({"litoria_aurea": 1, "non_target": 0})
CONFIG_SHA256 = "a" * 64


class BuildManifestRowsTests(unittest.TestCase):
    def test_maps_examples_through_plan_and_lowercases_digest(self) -> None:
        example = LabeledExample(
            example_id="recording_start0s",
            image_path="images/recording_start0s.png",
            label=1,
            label_name="litoria_aurea",
            recording_id="recording",
            start_s=0,
        )
        plan = SplitPlan(
            folds=5,
            seed=1337,
            test_fold=0,
            val_fold=1,
            fold_by_recording=MappingProxyType({"recording": 3}),
            split_by_recording=MappingProxyType({"recording": "train"}),
        )

        rows = build_manifest_rows((example,), plan, "ABCDEF" * 10 + "ABCD")

        self.assertEqual(rows, (ManifestRow(
            manifest_version=1,
            example_id=example.example_id,
            image_path=example.image_path,
            label=example.label,
            label_name=example.label_name,
            recording_id=example.recording_id,
            start_s=example.start_s,
            fold=3,
            split="train",
            preprocessing_config_sha256="abcdef" * 10 + "abcd",
        ),))


class ValidateManifestRowsTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.repo_root = Path(temporary_directory.name)
        self.rows = self._valid_rows()
        for row in self.rows:
            path = self.repo_root / row.image_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")

    def test_accepts_valid_completed_rows(self) -> None:
        self.assertIsNone(self.validate(self.rows))

    def test_reports_missing_file(self) -> None:
        missing = replace(self.rows[0], image_path="images/missing.png")

        self.assert_issue("missing_image_file", self._replace(0, missing))

    def test_reports_wrong_manifest_version(self) -> None:
        invalid = replace(self.rows[0], manifest_version=2)

        self.assert_issue("unsupported_manifest_version", self._replace(0, invalid))

    def test_reports_wrong_numeric_label(self) -> None:
        invalid = replace(self.rows[0], label=2)

        self.assert_issue("label_mapping_mismatch", self._replace(0, invalid))

    def test_reports_invalid_split(self) -> None:
        invalid = replace(self.rows[4], split="evaluation")

        self.assert_issue("invalid_split", self._replace(4, invalid))

    def test_reports_invalid_fold(self) -> None:
        invalid = replace(self.rows[8], fold=-1)

        self.assert_issue("invalid_fold", self._replace(8, invalid))

    def test_reports_invalid_sha256(self) -> None:
        invalid = replace(self.rows[0], preprocessing_config_sha256="NOT-A-SHA256")

        self.assert_issue(
            "invalid_preprocessing_config_sha256",
            self._replace(0, invalid),
        )

    def test_reports_duplicate_identity(self) -> None:
        duplicate = replace(
            self.rows[1],
            example_id=self.rows[0].example_id,
            recording_id=self.rows[0].recording_id,
            start_s=self.rows[0].start_s,
        )

        self.assert_issue("duplicate_example_identity", self._replace(1, duplicate))

    def test_reports_case_insensitive_duplicate_path(self) -> None:
        duplicate = replace(
            self.rows[1],
            image_path=self.rows[0].image_path.upper(),
        )

        self.assert_issue("duplicate_image_path", self._replace(1, duplicate))

    def test_reports_recording_fold_leakage(self) -> None:
        leaked = replace(
            self.rows[6],
            example_id=f"{self.rows[4].recording_id}_start5s",
            recording_id=self.rows[4].recording_id,
            start_s=5,
        )

        self.assert_issue("recording_fold_leakage", self._replace(6, leaked))

    def test_reports_recording_split_leakage(self) -> None:
        leaked = replace(
            self.rows[5],
            example_id=f"{self.rows[4].recording_id}_start5s",
            recording_id=self.rows[4].recording_id,
            start_s=5,
            split="val",
        )

        self.assert_issue("recording_split_leakage", self._replace(5, leaked))

    def test_reports_missing_fold_number(self) -> None:
        rows = tuple(row for row in self.rows if row.fold != 3)

        self.assert_issue("non_contiguous_folds", rows)

    def test_reports_fold_lacking_one_class(self) -> None:
        rows = tuple(
            row
            for row in self.rows
            if not (row.fold == 4 and row.label == 1)
        )

        self.assert_issue("fold_missing_class", rows)

    def test_reports_split_lacking_one_class(self) -> None:
        rows = tuple(
            row
            for row in self.rows
            if not (row.split == "test" and row.label == 1)
        )

        self.assert_issue("split_missing_class", rows)

    def assert_issue(self, expected_code: str, rows: tuple[ManifestRow, ...]) -> None:
        with self.assertRaises(ManifestValidationError) as raised:
            self.validate(rows)

        self.assertIn(expected_code, [issue.code for issue in raised.exception.issues])

    def validate(self, rows: tuple[ManifestRow, ...]) -> None:
        return validate_manifest_rows(
            rows,
            self.repo_root,
            CLASSES,
            CONFIG_SHA256,
        )

    def _replace(self, index: int, row: ManifestRow) -> tuple[ManifestRow, ...]:
        rows = list(self.rows)
        rows[index] = row
        return tuple(rows)

    @staticmethod
    def _valid_rows() -> tuple[ManifestRow, ...]:
        rows = []
        for fold in range(5):
            split = "test" if fold == 0 else "val" if fold == 1 else "train"
            for label_name, label in CLASSES.items():
                recording_id = f"{label_name}_{fold}"
                example_id = f"{recording_id}_start0s"
                rows.append(ManifestRow(
                    manifest_version=1,
                    example_id=example_id,
                    image_path=f"images/{example_id}.png",
                    label=label,
                    label_name=label_name,
                    recording_id=recording_id,
                    start_s=0,
                    fold=fold,
                    split=split,
                    preprocessing_config_sha256=CONFIG_SHA256,
                ))
        return tuple(rows)


if __name__ == "__main__":
    unittest.main()
