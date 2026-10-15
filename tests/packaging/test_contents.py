from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from frog_classifier.packaging import contents
from frog_classifier.packaging.contents import BuildInfo


INFO = BuildInfo(
    version="0.1.0",
    commit="0123456789abcdef0123456789abcdef01234567",
    uncommitted=False,
    built_at="2026-10-15T09:30:00Z",
    python="3.13.14",
    torch="2.14.1+cpu",
    model_run="resnet18-20261014T114432Z",
)


class NamesTests(unittest.TestCase):
    def test_the_zip_names_the_version_and_commit(self) -> None:
        self.assertEqual(contents.zip_name(INFO), "FrogCallScanner-0.1.0-0123456.zip")
        self.assertEqual(contents.zip_name(replace(INFO, uncommitted=True)), "FrogCallScanner-0.1.0-0123456-uncommitted.zip")

    def test_the_build_information_reads_back(self) -> None:
        self.assertEqual(json.loads(contents.build_info_json(INFO)), {
            "version": "0.1.0",
            "commit": INFO.commit,
            "uncommitted": False,
            "built_at": "2026-10-15T09:30:00Z",
            "python": "3.13.14",
            "torch": "2.14.1+cpu",
            "model_run": "resnet18-20261014T114432Z",
        })


class TextTests(unittest.TestCase):
    def test_the_launcher_runs_the_packaged_python_in_isolation(self) -> None:
        text = contents.launcher_text()

        self.assertNotIn("\n", text.replace("\r\n", ""))
        self.assertIn(
            r'"%~dp0app\python\python.exe" -I -m frog_classifier.app.launch '
            r'--app "%~dp0app\classifier_app\app.py" --models "%~dp0app\models" '
            r'--data "%LOCALAPPDATA%\Frog Call Scanner"',
            text,
        )
        self.assertTrue(text.rstrip().endswith("if errorlevel 1 pause"))

    def test_the_launcher_explains_a_start_from_inside_the_zip(self) -> None:
        text = contents.launcher_text()

        self.assertIn(r'if not exist "%~dp0app\python\python.exe" (', text)
        self.assertIn("Extract All", text)

    def test_the_instructions_cover_starting_results_updates_and_limits(self) -> None:
        text = contents.instructions_text(INFO)

        for expected in (
            "Frog Call Scanner 0.1.0 (build 0123456, 2026-10-15)",
            "Extract All",
            '"Start Frog Call Scanner"',
            r"%LOCALAPPDATA%\Frog Call Scanner",
            "resnet18-20261014T114432Z",
            "one site",
            "midnight and noon",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, text.replace("\r\n", " "))
        self.assertNotIn("\n", text.replace("\r\n", ""))
        self.assertLessEqual(max(len(line) for line in text.split("\r\n")), 76)

    def test_the_data_folder_stays_whole_so_it_can_be_pasted(self) -> None:
        lines = contents.instructions_text(INFO).split("\r\n")

        self.assertIn("   " + contents.DATA_FOLDER, lines)

    def test_uncommitted_instructions_say_so(self) -> None:
        self.assertIn("(build 0123456, uncommitted, 2026-10-15)", contents.instructions_text(replace(INFO, uncommitted=True)))


class TreeTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _files(self, *relative_paths: str) -> None:
        for relative_path in relative_paths:
            path = self.root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

    def test_files_only_compiling_needs_are_removed(self) -> None:
        self._files(
            "include/Python.h",
            "libs/python313.lib",
            "Scripts/streamlit.exe",
            "Lib/site-packages/torch/include/ATen/ATen.h",
            "Lib/site-packages/torch/share/cmake/Torch/TorchConfig.cmake",
            "Lib/site-packages/numpy/random/lib/npyrandom.lib",
            "Lib/site-packages/torch/lib/torch_cpu.dll",
            "Lib/site-packages/torch/__init__.py",
            "python.exe",
        )

        contents.prune(self.root)

        remaining = {path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(remaining, {
            "Lib/site-packages/torch/lib/torch_cpu.dll",
            "Lib/site-packages/torch/__init__.py",
            "python.exe",
        })

    def test_paths_over_the_budget_are_named_longest_first(self) -> None:
        folder = "a" * 100
        self._files(f"{folder}/{'b' * 59}", f"{folder}/{'c' * 60}", f"{folder}/{'d' * 61}")

        self.assertEqual(
            contents.paths_over_budget(self.root, 160),
            [f"{folder}\\{'d' * 61}", f"{folder}\\{'c' * 60}"],
        )
        self.assertEqual(contents.longest_path(self.root), 162)
