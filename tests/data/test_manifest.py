from __future__ import annotations

import csv
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from frog_classifier.data.manifest import (
    MANIFEST_COLUMNS,
    ManifestRow,
    discover_labeled_examples,
    load_manifest,
    serialize_manifest,
)
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


class ManifestCsvTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.repo_root = Path(temporary_directory.name)
        self.config = load_test_config(self.repo_root)
        self.rows = self._valid_rows(self.config.sha256)
        for row in self.rows:
            write_png(self.repo_root, row.image_path)

    def test_serializes_exact_header_utf8_and_deterministic_row_order(self) -> None:
        rows = (
            replace(
                self.rows[1],
                example_id="enregistrement_é_start0s",
                image_path="labeled/non_target/enregistrement_é_start0s.png",
                recording_id="enregistrement_é",
            ),
            self.rows[0],
        )

        payload = serialize_manifest(rows)

        self.assertEqual(
            payload.splitlines()[0].decode("utf-8"),
            ",".join(MANIFEST_COLUMNS),
        )
        self.assertIn("enregistrement_é".encode(), payload)
        parsed = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
        self.assertEqual(
            [row["image_path"] for row in parsed],
            [
                "labeled/litoria_aurea/litoria_aurea_0_start0s.png",
                "labeled/non_target/enregistrement_é_start0s.png",
            ],
        )

    def test_loads_a_serialized_manifest_round_trip(self) -> None:
        manifest_path = self.repo_root / "manifest.csv"
        manifest_path.write_bytes(serialize_manifest(tuple(reversed(self.rows))))

        loaded = load_manifest(manifest_path, self.repo_root, self.config)

        self.assertEqual(loaded, tuple(sorted(self.rows, key=lambda row: row.image_path)))

    def test_rejects_missing_or_extra_columns(self) -> None:
        for columns in (MANIFEST_COLUMNS[:-1], MANIFEST_COLUMNS + ("extra",)):
            with self.subTest(columns=columns):
                manifest_path = self._write_csv(columns, ())

                with self.assertRaises(ManifestValidationError) as raised:
                    load_manifest(manifest_path, self.repo_root, self.config)

                self.assertEqual(
                    [issue.code for issue in raised.exception.issues],
                    ["invalid_manifest_columns"],
                )

    def test_aggregates_invalid_integer_rows(self) -> None:
        records = [self._record(self.rows[0]), self._record(self.rows[1])]
        records[0]["manifest_version"] = "one"
        records[1]["fold"] = "first"
        manifest_path = self._write_csv(MANIFEST_COLUMNS, records)

        with self.assertRaises(ManifestValidationError) as raised:
            load_manifest(manifest_path, self.repo_root, self.config)

        self.assertEqual(
            [issue.code for issue in raised.exception.issues],
            ["invalid_manifest_integer", "invalid_manifest_integer"],
        )
        self.assertEqual(
            [issue.subject for issue in raised.exception.issues],
            ["row 2: manifest_version", "row 3: fold"],
        )

    def test_rejects_rows_with_missing_or_extra_values(self) -> None:
        header, row = serialize_manifest((self.rows[0],)).decode("utf-8").splitlines()
        malformed_rows = (row.rsplit(",", 1)[0], f"{row},extra")
        for malformed_row in malformed_rows:
            with self.subTest(malformed_row=malformed_row):
                manifest_path = self.repo_root / "malformed.csv"
                manifest_path.write_text(
                    f"{header}\n{malformed_row}\n",
                    encoding="utf-8",
                )

                with self.assertRaises(ManifestValidationError) as raised:
                    load_manifest(manifest_path, self.repo_root, self.config)

                self.assertIn(
                    "invalid_manifest_row",
                    [issue.code for issue in raised.exception.issues],
                )

    def test_rejects_unsupported_version_invalid_hash_and_missing_file(self) -> None:
        cases = (
            (replace(self.rows[0], manifest_version=2), "unsupported_manifest_version"),
            (
                replace(self.rows[0], preprocessing_config_sha256="not-a-hash"),
                "invalid_preprocessing_config_sha256",
            ),
            (replace(self.rows[0], image_path="images/missing.png"), "missing_image_file"),
        )
        for row, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                manifest_path = self.repo_root / f"{expected_code}.csv"
                all_rows = (row,) + self.rows[1:]
                manifest_path.write_bytes(serialize_manifest(all_rows))

                with self.assertRaises(ManifestValidationError) as raised:
                    load_manifest(manifest_path, self.repo_root, self.config)

                self.assertIn(
                    expected_code,
                    [issue.code for issue in raised.exception.issues],
                )

    def test_rejects_noncanonical_image_paths(self) -> None:
        canonical = self.rows[0]
        cases = (
            (
                replace(
                    canonical,
                    image_path=canonical.image_path.replace("/", "\\"),
                ),
                "invalid_image_path",
            ),
            (
                replace(canonical, image_path=canonical.image_path[:-4] + ".PNG"),
                "invalid_example_filename",
            ),
            (
                replace(canonical, image_path=canonical.image_path[:-4] + ".jpg"),
                "invalid_example_filename",
            ),
            (
                replace(
                    canonical,
                    image_path=canonical.image_path.replace(
                        "/litoria_aurea/",
                        "/non_target/",
                    ),
                ),
                "image_label_directory_mismatch",
            ),
            (
                replace(
                    canonical,
                    image_path=canonical.image_path.replace(
                        "labeled/litoria_aurea/",
                        "images/",
                    ),
                ),
                "image_label_directory_mismatch",
            ),
            (
                replace(
                    canonical,
                    image_path=canonical.image_path.replace(
                        canonical.example_id,
                        "different_recording_start0s",
                    ),
                ),
                "manifest_identity_mismatch",
            ),
        )
        for invalid, expected_code in cases:
            with self.subTest(image_path=invalid.image_path):
                self._assert_rejected_row(invalid, expected_code)

    def test_rejects_tampered_identity_and_chunk_start_fields(self) -> None:
        canonical = self.rows[0]
        cases = (
            replace(canonical, example_id="tampered_start0s"),
            replace(canonical, recording_id="tampered"),
            replace(canonical, start_s=5),
            replace(
                canonical,
                example_id=f"{canonical.recording_id}_start-5s",
                image_path=(
                    f"labeled/{canonical.label_name}/"
                    f"{canonical.recording_id}_start-5s.png"
                ),
                start_s=-5,
            ),
            replace(
                canonical,
                example_id=f"{canonical.recording_id}_start3s",
                image_path=(
                    f"labeled/{canonical.label_name}/"
                    f"{canonical.recording_id}_start3s.png"
                ),
                start_s=3,
            ),
        )
        for invalid in cases:
            with self.subTest(row=invalid):
                self._assert_rejected_row(invalid, "manifest_identity_mismatch")

    def _assert_rejected_row(
        self,
        row: ManifestRow,
        expected_code: str,
    ) -> None:
        image_path = self.repo_root / Path(row.image_path)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"")
        manifest_path = self.repo_root / "tampered.csv"
        manifest_path.write_bytes(serialize_manifest((row,) + self.rows[1:]))

        with self.assertRaises(ManifestValidationError) as raised:
            load_manifest(manifest_path, self.repo_root, self.config)

        self.assertIn(
            expected_code,
            [issue.code for issue in raised.exception.issues],
        )

    def _write_csv(
        self,
        columns: tuple[str, ...],
        records: tuple[dict[str, object], ...] | list[dict[str, object]],
    ) -> Path:
        path = self.repo_root / "input.csv"
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
        return path

    @staticmethod
    def _record(row: ManifestRow) -> dict[str, object]:
        return {
            column: getattr(row, column)
            for column in MANIFEST_COLUMNS
        }

    @staticmethod
    def _valid_rows(config_sha256: str) -> tuple[ManifestRow, ...]:
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
                    preprocessing_config_sha256=config_sha256,
                ))
        return tuple(rows)
