from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from frog_classifier.data.manifest import discover_labeled_examples
from frog_classifier.data.validation import ManifestIssue, ManifestValidationError
from tests.data.helpers import load_test_config, write_png


class DiscoverLabeledExamplesTests(unittest.TestCase):
    def discover(self, paths: tuple[str, ...]):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        repo_root = Path(temporary_directory.name)
        label_root = repo_root / "labeled"
        for relative_path in paths:
            write_png(label_root, relative_path)
        return discover_labeled_examples(
            label_root,
            repo_root,
            load_test_config(repo_root),
        )

    def test_discovers_canonical_labels_nested_paths_and_sorted_paths(self) -> None:
        examples = self.discover((
            "non_target/z/recording_b_start10s.png",
            "litoria_aurea/nested/recording_a_start5s.png",
        ))

        self.assertEqual(
            examples,
            (
                self._example(
                    "recording_a_start5s",
                    "labeled/litoria_aurea/nested/recording_a_start5s.png",
                    1,
                    "litoria_aurea",
                    "recording_a",
                    5,
                ),
                self._example(
                    "recording_b_start10s",
                    "labeled/non_target/z/recording_b_start10s.png",
                    0,
                    "non_target",
                    "recording_b",
                    10,
                ),
            ),
        )

    def test_reports_missing_label_directory(self) -> None:
        with self.assertRaises(ManifestValidationError) as raised:
            self.discover(("litoria_aurea/recording_start5s.png",))

        self.assertEqual(
            [issue.code for issue in raised.exception.issues],
            ["missing_label_directory"],
        )

    def test_reports_unknown_label_directory(self) -> None:
        with self.assertRaises(ManifestValidationError) as raised:
            self.discover((
                "litoria_aurea/recording_start5s.png",
                "non_target/other_start10s.png",
                "review/recording_start15s.png",
            ))

        self.assertEqual(
            [issue.code for issue in raised.exception.issues],
            ["unknown_label_directory"],
        )

    def test_reports_unparseable_filename(self) -> None:
        with self.assertRaises(ManifestValidationError) as raised:
            self.discover((
                "litoria_aurea/recording_start5s.png",
                "non_target/not-a-canonical-name.png",
            ))

        self.assertEqual(
            [issue.code for issue in raised.exception.issues],
            ["invalid_example_filename"],
        )

    def test_reports_duplicate_example_identity_across_classes(self) -> None:
        with self.assertRaises(ManifestValidationError) as raised:
            self.discover((
                "litoria_aurea/recording_start5s.png",
                "non_target/recording_start5s.png",
            ))

        self.assertEqual(
            [issue.code for issue in raised.exception.issues],
            ["duplicate_example_identity"],
        )

    def test_validation_error_sorts_issues(self) -> None:
        error = ManifestValidationError((
            ManifestIssue("z_code", "later", "z"),
            ManifestIssue("a_code", "first", "a"),
        ))

        self.assertEqual([issue.code for issue in error.issues], ["a_code", "z_code"])
        self.assertEqual(str(error), "[a_code] a: first\n[z_code] z: later")

    @staticmethod
    def _example(
        example_id: str,
        image_path: str,
        label: int,
        label_name: str,
        recording_id: str,
        start_s: int,
    ):
        from frog_classifier.data.manifest import LabeledExample

        return LabeledExample(
            example_id=example_id,
            image_path=image_path,
            label=label,
            label_name=label_name,
            recording_id=recording_id,
            start_s=start_s,
        )
