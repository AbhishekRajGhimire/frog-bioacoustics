from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "sync_labeled_images.py"


class SyncLabeledImagesTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.labeled_root = self.root / "labeled"
        self.spectrogram_root = self.root / "processed" / "spectrograms"
        self.module = self._load_module()
        self.labeled = self._write(self.labeled_root / "litoria_aurea" / "rec_start5s.png", b"old")
        self.fresh = self._write(self.spectrogram_root / "site" / "rec_start5s.png", b"new")
        self.queued = self._write(self.spectrogram_root / "site" / "rec_start10s.png", b"queue")

    def test_replaces_labeled_bytes_and_removes_the_queue_copy(self) -> None:
        result, output = self._run()

        self.assertEqual(result, 0)
        self.assertIn("Replaced 1 labeled image", output)
        self.assertEqual(self.labeled.read_bytes(), b"new")
        self.assertFalse(self.fresh.exists())
        self.assertEqual(self.queued.read_bytes(), b"queue")

    def test_dry_run_changes_nothing(self) -> None:
        result, output = self._run("--dry-run")

        self.assertEqual(result, 0)
        self.assertIn("Dry run", output)
        self.assertEqual(self.labeled.read_bytes(), b"old")
        self.assertTrue(self.fresh.exists())

    def test_missing_counterpart_fails_without_touching_anything(self) -> None:
        self._write(self.labeled_root / "non_target" / "other_start0s.png", b"old")

        result, output = self._run()

        self.assertEqual(result, 2)
        self.assertIn("[missing_counterpart]", output)
        self.assertIn("other_start0s.png", output)
        self.assertEqual(self.labeled.read_bytes(), b"old")
        self.assertTrue(self.fresh.exists())

    def test_ambiguous_counterpart_fails_without_touching_anything(self) -> None:
        self._write(self.spectrogram_root / "elsewhere" / "rec_start5s.png", b"duplicate")

        result, output = self._run()

        self.assertEqual(result, 2)
        self.assertIn("[ambiguous_counterpart]", output)
        self.assertEqual(self.labeled.read_bytes(), b"old")
        self.assertTrue(self.fresh.exists())

    def test_unparseable_labeled_name_fails_without_touching_anything(self) -> None:
        self._write(self.labeled_root / "non_target" / "notes.png", b"old")

        result, output = self._run()

        self.assertEqual(result, 2)
        self.assertIn("[invalid_example_filename]", output)
        self.assertEqual(self.labeled.read_bytes(), b"old")
        self.assertTrue(self.fresh.exists())

    def test_duplicate_labeled_names_fail_without_touching_anything(self) -> None:
        duplicate = self._write(self.labeled_root / "non_target" / "rec_start5s.png", b"other")

        result, output = self._run()

        self.assertEqual(result, 2)
        self.assertIn("[duplicate_labeled_name]", output)
        self.assertEqual(self.labeled.read_bytes(), b"old")
        self.assertEqual(duplicate.read_bytes(), b"other")
        self.assertTrue(self.fresh.exists())

    def test_stops_partway_on_replace_failure_and_reports_recovery(self) -> None:
        labeled2 = self._write(self.labeled_root / "non_target" / "rec2_start5s.png", b"old2")
        fresh2 = self._write(self.spectrogram_root / "site" / "rec2_start5s.png", b"new2")

        with patch.object(self.module.os, "replace", side_effect=[None, OSError("locked")]):
            result, output = self._run()

        self.assertEqual(result, 2)
        self.assertIn("Sync stopped after 1 of 2", output)
        self.assertIn("Run scripts/slice_audio.py", output)
        self.assertNotIn("Traceback", output)

    def test_nested_roots_are_rejected(self) -> None:
        result, output = self._run(
            "--labeled-root", str(self.spectrogram_root / "site"),
            "--spectrogram-root", str(self.spectrogram_root),
        )

        self.assertEqual(result, 2)
        self.assertIn("[nested_roots]", output)

    def _run(self, *extra: str) -> tuple[int, str]:
        argv = list(extra)
        if "--labeled-root" not in argv:
            argv += ["--labeled-root", str(self.labeled_root)]
        if "--spectrogram-root" not in argv:
            argv += ["--spectrogram-root", str(self.spectrogram_root)]
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = self.module.main(argv)
        return result, output.getvalue()

    def _write(self, path: Path, content: bytes) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _load_module(self):
        module_name = f"sync_labeled_images_test_{id(self)}"
        specification = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)
        specification.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
