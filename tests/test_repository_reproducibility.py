from __future__ import annotations

import re
import subprocess
import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

PLACEHOLDERS = (
    "raw/external/.gitkeep",
    "raw/ponds/.gitkeep",
    "processed/external/chunks/.gitkeep",
    "processed/external/spectrograms/.gitkeep",
    "processed/ponds/chunks/.gitkeep",
    "processed/ponds/spectrograms/.gitkeep",
    "labeled/litoria_aurea/.gitkeep",
    "labeled/non_target/.gitkeep",
    "labeled/unsure/.gitkeep",
    "models/.gitkeep",
    "results/analytics/.gitkeep",
    "results/data_quality/.gitkeep",
)

GENERATED_PATHS = (
    "raw/external/example.WAV",
    "processed/external/chunks/example.wav",
    "processed/external/spectrograms/example.png",
    "labeled/litoria_aurea/example.png",
    "models/example.pkl",
    "results/analytics/example.csv",
    "results/data_quality/example.json",
)

DATA_PACKAGE_MODULES = (
    "src/frog_classifier/data/__init__.py",
    "src/frog_classifier/data/config.py",
    "src/frog_classifier/data/manifest.py",
    "src/frog_classifier/data/naming.py",
    "src/frog_classifier/data/reporting.py",
    "src/frog_classifier/data/splitting.py",
    "src/frog_classifier/data/validation.py",
)

PREPROCESSING_PACKAGE_MODULES = (
    "src/frog_classifier/preprocessing/__init__.py",
    "src/frog_classifier/preprocessing/display.py",
    "src/frog_classifier/preprocessing/recording.py",
    "src/frog_classifier/preprocessing/spectrogram.py",
)

LABELING_PACKAGE_MODULES = (
    "src/frog_classifier/labeling/__init__.py",
    "src/frog_classifier/labeling/audio.py",
    "src/frog_classifier/labeling/decisions.py",
    "src/frog_classifier/labeling/queue.py",
)

OPERATIONAL_DOCUMENTS = (
    "README.md",
    "roadmap.md",
    "docs/outline.md",
    "docs/architecture.md",
    "docs/workflow.md",
    "docs/decisions/2026-09-13-spectrogram-rendering.md",
)

OBSOLETE_PATHS = (
    "PROCESS.md",
    "docs/system design.md",
    "dataset/train/litoria_aurea/.gitkeep",
    "dataset/train/non_target/.gitkeep",
    "dataset/val/litoria_aurea/.gitkeep",
    "dataset/val/non_target/.gitkeep",
    "dataset/test/litoria_aurea/.gitkeep",
    "dataset/test/non_target/.gitkeep",
    "scripts/label_spectrograms.py",
)

RUNTIME_DEPENDENCIES = {
    "librosa",
    "matplotlib",
    "numpy",
    "pillow",
    "scikit-learn",
    "simpleaudio",
    "streamlit",
    "tqdm",
}

EXPECTED_RUNTIME_REQUIREMENTS = {
    "librosa>=0.11,<0.12",
    "matplotlib>=3.10,<3.11",
    "numpy>=2.3,<2.4",
    "pillow>=12.1,<12.2",
    "scikit-learn>=1.8,<1.9",
    "simpleaudio==1.0.4",
    "streamlit>=1.54,<1.55",
    "tqdm>=4.67,<4.68",
}

OPERATIONAL_TEXT_FILES = (
    "README.md",
    "docs/outline.md",
    "docs/architecture.md",
    "docs/workflow.md",
    "scripts/build_manifest.py",
    "scripts/slice_audio.py",
    "scripts/sync_labeled_images.py",
    "scripts/train_baseline.py",
    "scripts/verify_spectrograms.py",
)

STALE_LAYOUT_TOKENS = ("Data/", "Data\\")

MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def git_path_is_ignored(relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", relative_path],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git check-ignore failed for {relative_path}: {result.returncode}")
    return result.returncode == 0


class RepositoryLayoutTests(unittest.TestCase):
    def test_generated_files_are_ignored_and_placeholders_are_trackable(self) -> None:
        for relative_path in GENERATED_PATHS:
            with self.subTest(generated=relative_path):
                self.assertTrue(git_path_is_ignored(relative_path))

        for relative_path in PLACEHOLDERS:
            with self.subTest(placeholder=relative_path):
                self.assertTrue((REPO_ROOT / relative_path).is_file())
                self.assertFalse(git_path_is_ignored(relative_path))

        for relative_path in OBSOLETE_PATHS:
            with self.subTest(obsolete=relative_path):
                self.assertFalse((REPO_ROOT / relative_path).exists())

    def test_project_navigation_files_are_complete_and_linked(self) -> None:
        for relative_path in (
            *OPERATIONAL_DOCUMENTS,
            "config/preprocessing.toml",
            *DATA_PACKAGE_MODULES,
            *PREPROCESSING_PACKAGE_MODULES,
            *LABELING_PACKAGE_MODULES,
        ):
            with self.subTest(required=relative_path):
                self.assertTrue((REPO_ROOT / relative_path).is_file())

        for relative_path in OPERATIONAL_DOCUMENTS:
            document_path = REPO_ROOT / relative_path
            document = document_path.read_text(encoding="utf-8")
            for target in MARKDOWN_LINK_PATTERN.findall(document):
                target_path = target.split("#", maxsplit=1)[0]
                if not target_path or "://" in target_path or target_path.startswith("mailto:"):
                    continue
                with self.subTest(document=relative_path, target=target):
                    self.assertTrue((document_path.parent / target_path).exists())

    def test_beginner_outline_is_discoverable(self) -> None:
        outline_path = REPO_ROOT / "docs/outline.md"
        self.assertTrue(outline_path.is_file())
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("[project outline](docs/outline.md)", readme)

    def test_text_and_binary_git_attributes_are_explicit(self) -> None:
        attributes_path = REPO_ROOT / ".gitattributes"
        self.assertTrue(attributes_path.is_file())
        attributes = attributes_path.read_text(encoding="utf-8")
        self.assertIn("* text=auto eol=lf", attributes)
        for pattern in ("*.docx binary", "*.png binary", "*.wav binary", "*.WAV binary", "*.xlsx binary"):
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, attributes)


class DependencyMetadataTests(unittest.TestCase):
    def test_dependency_metadata_and_compatibility_export_agree(self) -> None:
        pyproject_path = REPO_ROOT / "pyproject.toml"
        lock_path = REPO_ROOT / "uv.lock"
        python_version_path = REPO_ROOT / ".python-version"
        export_path = REPO_ROOT / "requirements_labeler.txt"

        self.assertTrue(pyproject_path.is_file())
        self.assertTrue(lock_path.is_file())
        self.assertEqual(python_version_path.read_text(encoding="utf-8").strip(), "3.13")

        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        self.assertEqual(pyproject["project"]["requires-python"], ">=3.13,<3.14")
        self.assertIn("link-mode", pyproject["tool"]["uv"])
        self.assertEqual(pyproject["tool"]["uv"]["link-mode"], "copy")
        self.assertEqual(set(pyproject["project"]["dependencies"]), EXPECTED_RUNTIME_REQUIREMENTS)
        direct_names = {
            re.split(r"[<>=!~\[]", requirement, maxsplit=1)[0].strip().lower()
            for requirement in pyproject["project"]["dependencies"]
        }
        self.assertEqual(direct_names, RUNTIME_DEPENDENCIES)

        exported_names = {
            line.split("==", maxsplit=1)[0].strip().lower()
            for line in export_path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith(("#", "-")) and "==" in line
        }
        self.assertTrue(RUNTIME_DEPENDENCIES.issubset(exported_names))

        launcher = (REPO_ROOT / "Launch_Labeler.bat").read_text(encoding="utf-8")
        self.assertIn("py -3.13", launcher)
        self.assertIn("Python 3.13", launcher)
        self.assertIn("requirements_labeler.txt", launcher)


class OperationalDocumentationTests(unittest.TestCase):
    def test_outline_requires_staged_diff_review_before_commit(self) -> None:
        outline_lines = (REPO_ROOT / "docs/outline.md").read_text(encoding="utf-8").splitlines()
        for command in (
            "git diff --cached --check",
            "git diff --cached",
            "git status --short",
        ):
            with self.subTest(command=command):
                self.assertIn(command, outline_lines)

    def test_operational_text_uses_the_canonical_layout(self) -> None:
        for relative_path in OPERATIONAL_TEXT_FILES:
            text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
            for stale_token in STALE_LAYOUT_TOKENS:
                with self.subTest(file=relative_path, token=stale_token):
                    self.assertNotIn(stale_token, text)

    def test_roadmap_covers_every_approved_phase(self) -> None:
        roadmap_path = REPO_ROOT / "roadmap.md"
        self.assertTrue(roadmap_path.is_file())
        roadmap = roadmap_path.read_text(encoding="utf-8")
        for phase in range(1, 8):
            with self.subTest(phase=phase):
                self.assertIn(f"## Phase {phase}:", roadmap)
        self.assertIn("Status: Next", roadmap)
        self.assertIn("Status: Planned", roadmap)

    def test_workflow_documents_every_operational_command(self) -> None:
        workflow = (REPO_ROOT / "docs/workflow.md").read_text(encoding="utf-8")
        for command in (
            "uv run python scripts/slice_audio.py",
            "uv run python scripts/sync_labeled_images.py",
            "uv run python scripts/verify_spectrograms.py --sample 50",
            "uv run python scripts/build_manifest.py",
            "uv run python scripts/train_baseline.py",
        ):
            with self.subTest(command=command):
                self.assertIn(command, workflow)


if __name__ == "__main__":
    unittest.main()
