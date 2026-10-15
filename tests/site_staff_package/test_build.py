from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from frog_classifier.packaging import build


def make_repository(repo: Path) -> None:
    """A committed repository with one packaged source file and one other file."""
    (repo / "src" / "frog_classifier").mkdir(parents=True)
    (repo / "src" / "frog_classifier" / "__init__.py").write_bytes(b"\n")
    (repo / "notes.txt").write_bytes(b"notes\n")
    for arguments in (
        ("init", "--quiet"),
        ("add", "."),
        ("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--quiet", "-m", "start"),
    ):
        subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True)


class SourceStateTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.repo = self.root / "repo"
        make_repository(self.repo)

    def test_a_committed_tree_is_not_uncommitted(self) -> None:
        commit, uncommitted = build.source_state(self.repo)

        self.assertEqual(len(commit), 40)
        self.assertFalse(uncommitted)

    def test_a_changed_packaged_source_is_uncommitted(self) -> None:
        (self.repo / "src" / "frog_classifier" / "__init__.py").write_bytes(b"changed\n")

        self.assertTrue(build.source_state(self.repo)[1])

    def test_a_new_packaged_source_is_uncommitted(self) -> None:
        (self.repo / "src" / "frog_classifier" / "new.py").write_bytes(b"x = 1\n")

        self.assertTrue(build.source_state(self.repo)[1])

    def test_changes_outside_the_packaged_sources_do_not_count(self) -> None:
        (self.repo / "notes.txt").write_bytes(b"changed\n")
        (self.repo / "scratch.txt").write_bytes(b"new\n")

        self.assertFalse(build.source_state(self.repo)[1])

    def test_an_uncommitted_build_is_refused_before_anything_is_made(self) -> None:
        (self.repo / "src" / "frog_classifier" / "__init__.py").write_bytes(b"changed\n")
        output = self.root / "dist"

        with self.assertRaisesRegex(build.BuildError, "--allow-uncommitted"):
            build.build_package(
                self.repo, self.root / "model", output,
                uv="uv", allow_uncommitted=False, now=datetime(2026, 10, 15, tzinfo=timezone.utc),
            )
        self.assertFalse(output.exists())


class InstalledSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.repo = root / "repo"
        make_repository(self.repo)
        (self.repo / "src" / "frog_classifier" / "app").mkdir()
        (self.repo / "src" / "frog_classifier" / "app" / "launch.py").write_bytes(b"new = True\n")
        (self.repo / "src" / "frog_classifier" / "__pycache__").mkdir()
        (self.repo / "src" / "frog_classifier" / "__pycache__" / "x.pyc").write_bytes(b"compiled")
        self.installed = root / "site-packages" / "frog_classifier"
        shutil.copytree(self.repo / "src" / "frog_classifier", self.installed)

    def test_an_installed_copy_of_the_source_matches_it(self) -> None:
        (self.installed / "__pycache__" / "y.pyc").write_bytes(b"compiled elsewhere")

        self.assertEqual(build.installed_source_differences(self.repo, self.installed), [])

    def test_stale_missing_and_leftover_installed_files_are_named(self) -> None:
        (self.installed / "app" / "launch.py").write_bytes(b"new = False\n")
        (self.repo / "src" / "frog_classifier" / "added.py").write_bytes(b"x = 1\n")
        (self.installed / "removed.py").write_bytes(b"y = 1\n")

        self.assertEqual(
            build.installed_source_differences(self.repo, self.installed),
            ["added.py", "removed.py", "app/launch.py"],
        )


class CommandTests(unittest.TestCase):
    def test_installing_python_leaves_the_computers_own_setup_alone(self) -> None:
        install_dir = Path("C:/staging/python-download")

        command = build.python_install_command("uv", install_dir)

        self.assertEqual(command[:4], ["uv", "python", "install", "3.13.14"])
        self.assertIn("--no-bin", command)
        self.assertIn("--no-registry", command)
        self.assertEqual(command[command.index("--install-dir") + 1], str(install_dir))

    def test_dependencies_come_from_the_lock_with_the_processor_pytorch(self) -> None:
        command = build.sync_command("uv", "frog-classifier")

        self.assertEqual(
            command[:7], ["uv", "sync", "--frozen", "--no-default-groups", "--group", "model-cpu", "--no-editable"],
        )

    def test_the_project_is_built_from_its_source_rather_than_an_earlier_build(self) -> None:
        # uv otherwise reuses its cached build of the project until pyproject.toml changes.
        command = build.sync_command("uv", "frog-classifier")

        self.assertEqual(command[command.index("--reinstall-package") + 1], "frog-classifier")

    def test_the_self_check_sees_only_what_windows_programs_need(self) -> None:
        environment = build.essential_environment({
            "SYSTEMROOT": "C:\\Windows",
            "TEMP": "C:\\Temp",
            "PATH": "C:\\Users\\me\\.local\\bin;C:\\Python313",
            "PYTHONPATH": "C:\\elsewhere",
            "UV": "C:\\Users\\me\\.local\\bin\\uv.exe",
            "VIRTUAL_ENV": "D:\\repo\\.venv",
        })

        self.assertEqual(environment, {
            "SYSTEMROOT": "C:\\Windows",
            "TEMP": "C:\\Temp",
            "PATH": "C:\\Windows\\System32;C:\\Windows;C:\\Windows\\System32\\Wbem",
        })

    def test_uv_comes_from_uv_run(self) -> None:
        self.assertEqual(build.find_uv({"UV": "C:/uv/uv.exe"}), "C:/uv/uv.exe")
        with patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(build.BuildError, "uv run"):
                build.find_uv({})
