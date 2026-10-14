from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tests.inference.helpers import AUDIOMOTH_COMMENT, build_saved_run, write_recording
from tests.preprocessing.helpers import noise


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "detect_calls.py"


def _load(repo_root: Path):
    specification = importlib.util.spec_from_file_location("detect_calls_under_test", SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    # The command finds the repository from its own location.
    module.__file__ = str(repo_root / "scripts" / "detect_calls.py")
    return module


def _run(module, *arguments: str) -> tuple[int, str, str]:
    output, errors = io.StringIO(), io.StringIO()
    with patch.object(sys, "argv", ["detect_calls.py", *arguments]), redirect_stdout(output), redirect_stderr(errors):
        code = module.main()
    return code, output.getvalue(), errors.getvalue()


class DetectCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo = Path(directory.name)
        self.run_dir = build_saved_run(self.repo / "models")
        site = self.repo / "raw" / "site"
        write_recording(site / "20220123_010000.WAV", noise(10, sample_rate_hz=48000), comment=AUDIOMOTH_COMMENT)
        write_recording(site / "20220123_020000.WAV", noise(5, sample_rate_hz=48000, seed=1))
        self.command = _load(self.repo)
        self.arguments = (f"models/{self.run_dir.name}", "raw/site", "--workers", "1", "--device", "cpu")

    def test_detects_into_the_default_output_folder_and_summarizes(self) -> None:
        code, output, _ = _run(self.command, *self.arguments)

        self.assertEqual(code, 0)
        self.assertIn("2 recordings to process; 0 already done.", output)
        self.assertIn("Recordings: 2 found, 0 already done, 2 processed now, 0 failed.", output)
        self.assertIn("of 3 windows at or above the threshold 0.500.", output)
        self.assertIn("Speed:", output)
        output_dir = (self.repo / "results" / "detections" / f"site-{self.run_dir.name}").resolve()
        self.assertTrue((output_dir / "detections.csv").is_file())
        self.assertIn(f"Results: {output_dir}", output)

    def test_a_second_run_processes_nothing_new_and_rewrites_the_spreadsheets(self) -> None:
        _run(self.command, *self.arguments)
        output_dir = self.repo / "results" / "detections" / f"site-{self.run_dir.name}"
        (output_dir / "recordings.csv").unlink()

        code, output, _ = _run(self.command, *self.arguments)

        self.assertEqual(code, 0)
        self.assertIn("Recordings: 2 found, 2 already done, 0 processed now, 0 failed.", output)
        self.assertNotIn("Speed:", output)
        self.assertTrue((output_dir / "recordings.csv").is_file())

    def test_a_failed_recording_exits_with_one_and_says_how_to_retry(self) -> None:
        (self.repo / "raw" / "site" / "20220123_030000.wav").write_bytes(b"not audio")

        with warnings.catch_warnings():
            # librosa warns before trying its fallback decoder; the failure is what is under test.
            warnings.simplefilter("ignore")
            code, output, errors = _run(self.command, *self.arguments)

        self.assertEqual(code, 1)
        self.assertIn("FAILED 20220123_030000.wav", output)
        self.assertIn("Run the same command again to retry them.", errors)

    def test_a_folder_that_is_not_a_run_cannot_start(self) -> None:
        code, _, errors = _run(self.command, "models", "raw/site", "--workers", "1", "--device", "cpu")

        self.assertEqual(code, 2)
        self.assertIn("Cannot start:", errors)
        self.assertIn("is not a saved run", errors)


if __name__ == "__main__":
    unittest.main()
