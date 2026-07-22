from __future__ import annotations

import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from frog_classifier.data.validation import ManifestValidationError
from tests.data.helpers import VALID_CONFIG, write_png


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_manifest.py"


class BuildManifestCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory(
            dir=REPO_ROOT,
            prefix="manifest-cli-test-",
        )
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.training_root = self.root / "labeled"
        self.csv_path = self.root / "outputs" / "manifest.csv"
        self.report_dir = self.root / "reports"
        self.config_path = self.root / "config" / "preprocessing.toml"
        self.config_path.parent.mkdir(parents=True)
        self.config_path.write_bytes(VALID_CONFIG)

    def test_generates_three_byte_stable_outputs_with_both_classes_per_fold(self) -> None:
        for fold_group in range(5):
            for label_name in ("litoria_aurea", "non_target"):
                recording_id = f"{label_name}_{fold_group}"
                write_png(
                    self.training_root,
                    f"{label_name}/{recording_id}_start0s.png",
                )

        first = self._run()

        self.assertEqual(first.returncode, 0, first.stderr)
        output_paths = (
            self.csv_path,
            self.report_dir / "manifest-report.json",
            self.report_dir / "manifest-report.md",
        )
        for output_path in output_paths:
            self.assertTrue(output_path.is_file(), output_path)
            self.assertIn(str(output_path), first.stdout)
        first_payloads = {
            path: path.read_bytes()
            for path in output_paths
        }
        with self.csv_path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        for fold in range(5):
            self.assertEqual(
                {int(row["label"]) for row in rows if int(row["fold"]) == fold},
                {0, 1},
            )
        report = json.loads(
            (self.report_dir / "manifest-report.json").read_text(encoding="utf-8")
        )
        for fold in range(5):
            self.assertEqual(
                report["by_fold"][str(fold)]["examples_by_class"],
                {"litoria_aurea": 1, "non_target": 1},
            )

        second = self._run()

        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(
            {path: path.read_bytes() for path in output_paths},
            first_payloads,
        )

    def test_reports_all_discovery_issues_and_keeps_existing_outputs(self) -> None:
        write_png(
            self.training_root,
            "litoria_aurea/target_recording_start0s.png",
        )
        write_png(
            self.training_root,
            "non_target/not-a-canonical-name.png",
        )
        write_png(
            self.training_root,
            "review/unknown_recording_start0s.png",
        )
        expected_payloads = {
            self.csv_path: b"existing csv",
            self.report_dir / "manifest-report.json": b"existing json",
            self.report_dir / "manifest-report.md": b"existing markdown",
        }
        for path, payload in expected_payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

        completed = self._run()

        self.assertEqual(completed.returncode, 2)
        self.assertIn("unknown_label_directory", completed.stderr)
        self.assertIn("invalid_example_filename", completed.stderr)
        self.assertEqual(
            {path: path.read_bytes() for path in expected_payloads},
            expected_payloads,
        )

    def test_rejects_csv_collisions_with_each_report_without_replacing_outputs(
        self,
    ) -> None:
        for fold_group in range(5):
            for label_name in ("litoria_aurea", "non_target"):
                recording_id = f"{label_name}_{fold_group}"
                write_png(
                    self.training_root,
                    f"{label_name}/{recording_id}_start0s.png",
                )
        report_json = self.report_dir / "manifest-report.json"
        report_markdown = self.report_dir / "manifest-report.md"
        expected_payloads = {
            self.csv_path: b"existing csv",
            report_json: b"existing json",
            report_markdown: b"existing markdown",
        }

        for collision_path in (report_json, report_markdown):
            with self.subTest(collision_path=collision_path):
                for path, payload in expected_payloads.items():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(payload)

                completed = self._run(out_csv=collision_path)

                self.assertEqual(completed.returncode, 2)
                self.assertIn("output_path_collision", completed.stderr)
                self.assertIn(str(collision_path), completed.stderr)
                self.assertEqual(
                    {path: path.read_bytes() for path in expected_payloads},
                    expected_payloads,
                )

    def test_rejects_a_labeled_png_destination_without_replacing_it(self) -> None:
        self._write_valid_label_tree()
        protected_image = next(
            (self.training_root / "litoria_aurea").glob("*.png")
        )
        protected_image.write_bytes(b"human label")

        completed = self._run(out_csv=protected_image)

        self.assertEqual(completed.returncode, 2)
        self.assertIn("invalid_output_suffix", completed.stderr)
        self.assertIn("protected_output_path", completed.stderr)
        self.assertEqual(protected_image.read_bytes(), b"human label")
        self.assertFalse((self.report_dir / "manifest-report.json").exists())
        self.assertFalse((self.report_dir / "manifest-report.md").exists())

    def test_rejects_non_csv_destination_without_replacing_it(self) -> None:
        self._write_valid_label_tree()
        invalid_destination = self.root / "outputs" / "manifest.txt"
        invalid_destination.parent.mkdir(parents=True)
        invalid_destination.write_bytes(b"existing")

        completed = self._run(out_csv=invalid_destination)

        self.assertEqual(completed.returncode, 2)
        self.assertIn("invalid_output_suffix", completed.stderr)
        self.assertEqual(invalid_destination.read_bytes(), b"existing")

    def _write_valid_label_tree(self) -> None:
        for fold_group in range(5):
            for label_name in ("litoria_aurea", "non_target"):
                recording_id = f"{label_name}_{fold_group}"
                write_png(
                    self.training_root,
                    f"{label_name}/{recording_id}_start0s.png",
                )

    def _run(
        self,
        *,
        out_csv: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            (
                sys.executable,
                str(SCRIPT_PATH),
                "--training-root",
                str(self.training_root),
                "--out-csv",
                str(out_csv or self.csv_path),
                "--report-dir",
                str(self.report_dir),
                "--config",
                str(self.config_path),
                "--seed",
                "2026",
                "--folds",
                "5",
                "--test-fold",
                "0",
                "--val-fold",
                "1",
            ),
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )


class OutputPathValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.repo_root = Path(temporary_directory.name).resolve()
        self.training_root = (self.repo_root / "labeled").resolve()
        self.config_path = (
            self.repo_root / "config" / "preprocessing.toml"
        ).resolve()
        self.report_json = (
            self.repo_root / "results" / "data_quality" / "manifest-report.json"
        ).resolve()
        self.report_markdown = self.report_json.with_suffix(".md")
        self.module = self._load_script_module()

    def test_rejects_protected_roots_and_input_collisions(self) -> None:
        discovered_image = (
            self.training_root / "litoria_aurea" / "recording_start0s.png"
        ).resolve()
        cases = (
            (self.repo_root / "raw" / "manifest.csv", self.report_json),
            (self.repo_root / "processed" / "manifest.csv", self.report_json),
            (self.repo_root / "config" / "manifest.csv", self.report_json),
            (self.training_root / "replacement.csv", self.report_json),
            (self.repo_root / "outputs" / "manifest.csv", self.config_path),
            (discovered_image, self.report_json),
            (
                self.repo_root / "outputs" / "manifest.csv",
                self.repo_root / "raw" / "manifest-report.json",
            ),
        )
        for out_csv, report_json in cases:
            with self.subTest(out_csv=out_csv, report_json=report_json):
                with self.assertRaises(ManifestValidationError) as raised:
                    self.module._validate_output_paths(
                        out_csv=out_csv.resolve(),
                        out_csv_lexical=out_csv.absolute(),
                        report_paths=(
                            report_json.resolve(),
                            self.report_markdown,
                        ),
                        repo_root=self.repo_root,
                        training_root=self.training_root,
                        config_path=self.config_path,
                        discovered_image_paths=(discovered_image,),
                    )

                self.assertIn(
                    "protected_output_path",
                    [issue.code for issue in raised.exception.issues],
                )

    def test_allows_the_default_manifest_inside_the_canonical_label_tree(self) -> None:
        self.assertIsNone(self.module._validate_output_paths(
            out_csv=(self.repo_root / "labeled" / "manifest.csv").resolve(),
            out_csv_lexical=(
                self.repo_root / "labeled" / "manifest.csv"
            ).absolute(),
            report_paths=(self.report_json, self.report_markdown),
            repo_root=self.repo_root,
            training_root=self.training_root,
            config_path=self.config_path,
            discovered_image_paths=(),
        ))

    def test_symlinked_default_manifest_cannot_replace_protected_target(self) -> None:
        self.config_path.parent.mkdir(parents=True)
        self.config_path.write_bytes(VALID_CONFIG)
        for fold_group in range(5):
            for label_name in ("litoria_aurea", "non_target"):
                recording_id = f"{label_name}_{fold_group}"
                write_png(
                    self.training_root,
                    f"{label_name}/{recording_id}_start0s.png",
                )
        protected_target = self.repo_root / "external" / "protected.csv"
        protected_target.parent.mkdir(parents=True)
        protected_target.write_bytes(b"protected source")
        default_manifest = self.training_root / "manifest.csv"
        try:
            default_manifest.symlink_to(protected_target)
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"file symlink creation unavailable: {error}")

        self.module.__file__ = str(
            self.repo_root / "scripts" / "build_manifest.py"
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = self.module.main(())

        self.assertEqual(result, 2)
        self.assertIn("protected_output_path", stderr.getvalue())
        self.assertEqual(protected_target.read_bytes(), b"protected source")

    def _load_script_module(self):
        specification = importlib.util.spec_from_file_location(
            f"build_manifest_test_{id(self)}",
            SCRIPT_PATH,
        )
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
