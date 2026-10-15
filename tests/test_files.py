from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frog_classifier.files import FileInUseError, replace_file


class ReplaceFileTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.target = self.root / "report.md"
        self.target.write_bytes(b"old")

    def test_the_new_content_replaces_the_old(self) -> None:
        replace_file(self.target, lambda staging: staging.write_bytes(b"new"))

        self.assertEqual(self.target.read_bytes(), b"new")
        self.assertEqual([path.name for path in self.root.iterdir()], ["report.md"])

    def test_a_file_that_cannot_be_replaced_keeps_its_content_and_explains_why(self) -> None:
        with patch("frog_classifier.files.os.replace", side_effect=PermissionError(13, "Permission denied")):
            with self.assertRaisesRegex(
                FileInUseError, "report.md could not be replaced.*close it and run the same command again"
            ):
                replace_file(self.target, lambda staging: staging.write_bytes(b"new"))

        self.assertEqual(self.target.read_bytes(), b"old")
        self.assertEqual([path.name for path in self.root.iterdir()], ["report.md"])


if __name__ == "__main__":
    unittest.main()
