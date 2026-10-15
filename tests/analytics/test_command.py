from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tests.analytics.helpers import build_record, recording


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_activity.py"


def _load(repo_root: Path):
    specification = importlib.util.spec_from_file_location("analyze_activity_under_test", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    # The command finds the repository from its own location.
    module.__file__ = str(repo_root / "scripts" / "analyze_activity.py")
    return module


def _run(module, *arguments: str) -> tuple[int, str, str]:
    output, errors = io.StringIO(), io.StringIO()
    with patch.object(sys, "argv", ["analyze_activity.py", *arguments]), redirect_stdout(output), redirect_stderr(errors):
        code = module.main()
    return code, output.getvalue(), errors.getvalue()


class AnalyzeCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo = Path(directory.name)
        build_record(self.repo / "results" / "detections" / "site-run", [
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.95, 0.1, 0.6)),
        ])
        self.command = _load(self.repo)

    def test_analyzes_into_the_default_folder_and_summarizes(self) -> None:
        code, output, _ = _run(self.command, "results/detections/site-run")

        self.assertEqual(code, 0)
        self.assertIn("Calls were detected in 66.7% of recorded time", output)
        report = self.repo / "results" / "analytics" / "site-run" / "report.md"
        self.assertTrue(report.is_file())
        self.assertIn(f"Report: {report}", output)

    def test_overrides_reach_the_analysis(self) -> None:
        code, output, _ = _run(
            self.command, "results/detections/site-run",
            "--threshold", "0.9", "--utc-offset", "+10:00", "--output", "results/analytics/custom",
        )

        self.assertEqual(code, 0)
        self.assertIn("at threshold 0.900", output)
        report = (self.repo / "results" / "analytics" / "custom" / "report.md").read_bytes().decode("utf-8")
        self.assertIn("| Time basis | UTC+10:00 |", report)

    def test_a_folder_without_a_record_cannot_be_analyzed(self) -> None:
        code, _, errors = _run(self.command, "results/detections/missing")

        self.assertEqual(code, 2)
        self.assertIn("Cannot analyze:", errors)

    def test_a_malformed_offset_cannot_be_analyzed(self) -> None:
        code, _, errors = _run(self.command, "results/detections/site-run", "--utc-offset", "11")

        self.assertEqual(code, 2)
        self.assertIn("a UTC offset looks like +11:00", errors)

    def test_a_report_open_in_another_program_is_explained(self) -> None:
        with patch("frog_classifier.files.os.replace", side_effect=PermissionError(13, "Permission denied")):
            code, _, errors = _run(self.command, "results/detections/site-run")

        self.assertEqual(code, 1)
        self.assertIn("could not be replaced", errors)


if __name__ == "__main__":
    unittest.main()
