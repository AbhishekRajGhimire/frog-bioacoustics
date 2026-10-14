from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from frog_classifier.training.artifact import PREDICTIONS_FILE
from tests.training.helpers import build_synthetic_repo


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(script: str, repo_root: Path):
    name = f"{script}_under_test"
    specification = importlib.util.spec_from_file_location(name, SCRIPTS / f"{script}.py")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    # The commands find the repository from their own location.
    module.__file__ = str(repo_root / "scripts" / f"{script}.py")
    return module


def _run(module, *arguments: str) -> tuple[int, str]:
    output = io.StringIO()
    with patch.object(sys, "argv", [module.__name__, *arguments]), redirect_stdout(output):
        return module.main(), output.getvalue()


class CommandTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo = Path(directory.name)
        build_synthetic_repo(self.repo)

    def test_train_then_verify_from_the_command_line(self) -> None:
        code, output = _run(_load("train_model", self.repo), "--no-cross-validation", "--device", "cpu")

        self.assertEqual(code, 0)
        self.assertIn("Saved run:", output)
        (run_dir,) = (self.repo / "models").iterdir()
        self.assertFalse((run_dir / "cross_validation.csv").exists())

        verify = _load("verify_model", self.repo)
        code, output = _run(verify, str(run_dir.relative_to(self.repo)), "--device", "cpu")
        self.assertEqual(code, 0)
        self.assertIn("PASSED", output)

        predictions = run_dir / PREDICTIONS_FILE
        predictions.write_bytes(predictions.read_bytes() + b"\n")
        code, output = _run(verify, str(run_dir), "--device", "cpu")
        self.assertEqual(code, 1)
        self.assertIn("FAIL predictions.csv: checksum differs from run.json", output)


if __name__ == "__main__":
    unittest.main()
