from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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

    def _run(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            (
                sys.executable,
                str(SCRIPT_PATH),
                "--training-root",
                str(self.training_root),
                "--out-csv",
                str(self.csv_path),
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


if __name__ == "__main__":
    unittest.main()
