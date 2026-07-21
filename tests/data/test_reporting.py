from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from types import MappingProxyType
from typing import cast

from frog_classifier.data.manifest import ManifestRow, serialize_manifest
from frog_classifier.data.reporting import (
    PROVENANCE_NOTE,
    build_data_quality_report,
    serialize_report_json,
    serialize_report_markdown,
    write_output_bundle,
)
from frog_classifier.data.splitting import SplitPlan
from tests.data.helpers import load_test_config


class DataQualityReportTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.repo_root = Path(temporary_directory.name)
        self.config = load_test_config(self.repo_root)
        self.rows = self._rows(self.config.sha256)
        self.manifest_payload = serialize_manifest(self.rows)
        fold_by_recording = {
            row.recording_id: row.fold
            for row in self.rows
        }
        split_by_recording = {
            row.recording_id: row.split
            for row in self.rows
        }
        self.plan = SplitPlan(
            folds=5,
            seed=4242,
            test_fold=0,
            val_fold=1,
            fold_by_recording=MappingProxyType(fold_by_recording),
            split_by_recording=MappingProxyType(split_by_recording),
        )

    def test_builds_exact_example_and_group_counts(self) -> None:
        report = self._report()

        self.assertEqual(report.total.examples, 12)
        self.assertEqual(report.total.recording_groups, 10)
        self.assertEqual(
            {
                key: (value.examples, value.recording_groups)
                for key, value in report.by_class.items()
            },
            {"litoria_aurea": (6, 5), "non_target": (6, 5)},
        )
        self.assertEqual(
            {
                key: (value.examples, value.recording_groups)
                for key, value in report.by_fold.items()
            },
            {
                "0": (3, 2),
                "1": (2, 2),
                "2": (3, 2),
                "3": (2, 2),
                "4": (2, 2),
            },
        )
        self.assertEqual(
            {
                key: (value.examples, value.recording_groups)
                for key, value in report.by_split.items()
            },
            {"test": (3, 2), "train": (7, 6), "val": (2, 2)},
        )
        self.assertEqual(
            report.by_fold["0"].examples_by_class,
            {"litoria_aurea": 2, "non_target": 1},
        )
        self.assertEqual(
            report.by_fold["2"].recording_groups_by_class,
            {"litoria_aurea": 1, "non_target": 1},
        )
        self.assertEqual(
            report.by_split["train"].examples_by_class,
            {"litoria_aurea": 3, "non_target": 4},
        )
        self.assertEqual(
            report.by_split["train"].recording_groups_by_class,
            {"litoria_aurea": 3, "non_target": 3},
        )

    def test_records_reproducibility_validation_and_provenance(self) -> None:
        report = self._report()

        self.assertEqual(report.report_schema_version, 1)
        self.assertEqual(report.manifest_version, 1)
        self.assertEqual(report.seed, 4242)
        self.assertEqual(report.folds, 5)
        self.assertEqual(report.test_fold, 0)
        self.assertEqual(report.val_fold, 1)
        self.assertEqual(
            report.preprocessing_config_path,
            "config/preprocessing.toml",
        )
        self.assertEqual(
            report.preprocessing_config_sha256,
            self.config.sha256,
        )
        self.assertEqual(len(report.manifest_sha256), 64)
        self.assertEqual(
            [check.name for check in report.validation_checks],
            [
                "canonical_manifest_schema",
                "preprocessing_configuration_matches",
                "all_image_files_exist",
                "recording_groups_are_isolated",
                "every_fold_contains_all_classes",
                "every_split_contains_all_classes",
            ],
        )
        self.assertTrue(all(check.passed for check in report.validation_checks))
        self.assertEqual(report.provenance_note, PROVENANCE_NOTE)
        with self.assertRaises(FrozenInstanceError):
            report.seed = 1  # type: ignore[misc]

    def test_json_and_markdown_are_stable_and_contain_no_timestamp(self) -> None:
        report = self._report()

        first_json = serialize_report_json(report)
        second_json = serialize_report_json(report)
        first_markdown = serialize_report_markdown(report)
        second_markdown = serialize_report_markdown(report)

        self.assertEqual(first_json, second_json)
        self.assertEqual(first_markdown, second_markdown)
        expected_json = (
            json.dumps(
                asdict(report),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        self.assertEqual(first_json, expected_json)
        for payload in (first_json, first_markdown):
            self.assertNotIn(b"timestamp", payload.lower())
            self.assertNotIn(b"generated_at", payload.lower())
        self.assertIn(b"# Manifest Data Quality Report", first_markdown)
        self.assertIn(b"## Counts by Fold", first_markdown)
        self.assertIn(self.config.sha256.encode(), first_markdown)
        self.assertIn(PROVENANCE_NOTE.encode(), first_markdown)

    def _report(self):
        return build_data_quality_report(
            self.rows,
            self.plan,
            self.config,
            self.manifest_payload,
            self.repo_root,
        )

    @staticmethod
    def _rows(config_sha256: str) -> tuple[ManifestRow, ...]:
        rows = []
        for fold in range(5):
            split = "test" if fold == 0 else "val" if fold == 1 else "train"
            for label_name, label in (("litoria_aurea", 1), ("non_target", 0)):
                recording_id = f"{label_name}_{fold}"
                chunk_count = 2 if (
                    (fold == 0 and label_name == "litoria_aurea")
                    or (fold == 2 and label_name == "non_target")
                ) else 1
                for chunk in range(chunk_count):
                    start_s = chunk * 5
                    example_id = f"{recording_id}_start{start_s}s"
                    rows.append(ManifestRow(
                        manifest_version=1,
                        example_id=example_id,
                        image_path=f"images/{example_id}.png",
                        label=label,
                        label_name=label_name,
                        recording_id=recording_id,
                        start_s=start_s,
                        fold=fold,
                        split=split,
                        preprocessing_config_sha256=config_sha256,
                    ))
        return tuple(rows)


class WriteOutputBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)

    def test_replaces_all_destinations_after_preparing_temporary_files(self) -> None:
        first = self.root / "output" / "manifest.csv"
        second = self.root / "reports" / "manifest-report.json"
        first.parent.mkdir()
        second.parent.mkdir()
        first.write_bytes(b"old csv")
        second.write_bytes(b"old json")

        write_output_bundle({first: b"new csv", second: b"new json"})

        self.assertEqual(first.read_bytes(), b"new csv")
        self.assertEqual(second.read_bytes(), b"new json")
        self.assertEqual(list(self.root.rglob("*.tmp")), [])

    def test_preparation_failure_keeps_outputs_and_unrelated_temporary_files(self) -> None:
        valid_parent = self.root / "a_valid"
        valid_parent.mkdir()
        existing = valid_parent / "manifest.csv"
        existing.write_bytes(b"existing")
        unrelated = valid_parent / ".unrelated.tmp"
        unrelated.write_bytes(b"keep")
        blocked_parent = self.root / "z_blocked"
        blocked_parent.write_bytes(b"not a directory")

        with self.assertRaises(OSError):
            write_output_bundle({
                existing: b"replacement",
                blocked_parent / "report.json": b"report",
            })

        self.assertEqual(existing.read_bytes(), b"existing")
        self.assertEqual(unrelated.read_bytes(), b"keep")
        self.assertEqual(
            sorted(path.name for path in valid_parent.iterdir()),
            [".unrelated.tmp", "manifest.csv"],
        )

    def test_write_failure_removes_the_current_temporary_file(self) -> None:
        destination = self.root / "output" / "manifest.csv"

        with self.assertRaises(TypeError):
            write_output_bundle({destination: cast(bytes, object())})

        self.assertEqual(list(destination.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
