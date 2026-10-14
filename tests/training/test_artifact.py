from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from frog_classifier.training.artifact import (
    RUN_METADATA,
    check_files,
    finish_run,
    read_csv,
    read_run,
    start_run,
    write_csv,
)


class RunFolderTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.models = Path(directory.name) / "models"

    def test_a_run_is_staged_under_a_hidden_name_then_renamed_with_checksums(self) -> None:
        staging = start_run(self.models, "resnet18-x")
        (staging / "a.txt").write_bytes(b"alpha")

        final = finish_run(staging, {"run_name": "resnet18-x"})

        self.assertEqual(final, self.models / "resnet18-x")
        self.assertFalse(staging.exists())
        metadata = read_run(final)
        self.assertEqual(metadata["run_name"], "resnet18-x")
        self.assertEqual(metadata["files"], {"a.txt": hashlib.sha256(b"alpha").hexdigest()})
        self.assertEqual(check_files(final, metadata), [])
        self.assertNotIn(b"\r", (final / RUN_METADATA).read_bytes())
        self.assertEqual(json.loads((final / RUN_METADATA).read_text(encoding="utf-8")), metadata)

    def test_refuses_a_name_that_is_already_used_or_staged(self) -> None:
        (self.models / "resnet18-x").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            start_run(self.models, "resnet18-x")

        start_run(self.models, "resnet18-y")
        with self.assertRaises(FileExistsError):
            start_run(self.models, "resnet18-y")

    def test_reports_changed_and_missing_files(self) -> None:
        staging = start_run(self.models, "resnet18-x")
        (staging / "a.txt").write_bytes(b"alpha")
        (staging / "b.txt").write_bytes(b"beta")
        final = finish_run(staging, {})
        (final / "a.txt").write_bytes(b"changed")
        (final / "b.txt").unlink()

        problems = check_files(final, read_run(final))

        self.assertEqual(problems, [
            "a.txt: checksum differs from run.json",
            "b.txt: missing from the run folder",
        ])

    def test_csv_round_trip_uses_unix_line_endings(self) -> None:
        path = self.models / "rows.csv"
        self.models.mkdir()

        write_csv(path, ("name", "value"), [{"name": "a", "value": 1}, {"name": "b", "value": 0.5}])

        self.assertNotIn(b"\r", path.read_bytes())
        self.assertEqual(read_csv(path), [{"name": "a", "value": "1"}, {"name": "b", "value": "0.5"}])


if __name__ == "__main__":
    unittest.main()
