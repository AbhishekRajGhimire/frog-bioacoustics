# Repository Reproducibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a fresh clone reproduce the intended directory structure, Python environment, commands, documentation, and class vocabulary without changing classifier behavior or local data.

**Architecture:** Keep the project as a dependency-only collection of runnable Python scripts.
Use `pyproject.toml` for direct dependency policy, `uv.lock` for the exact environment, and a generated `requirements_labeler.txt` for the existing pip-based Windows launcher.
Protect the contract with repository-level `unittest` checks that exercise real Git ignore behavior and real project files.

**Tech Stack:** Python 3.13, standard-library `unittest`, Git ignore rules, uv 0.11, pip-compatible requirements export, Markdown, Windows batch.

## Global Constraints

- Support Python `>=3.13,<3.14` and record `3.13` in `.python-version`.
- Use `litoria_aurea` and `non_target` as the canonical filesystem and manifest class names.
- Do not modify, move, or delete raw recordings, generated spectrograms, cached audio chunks, or labeled images.
- Do not change manifest splitting, model training, inference, or analytics behavior.
- Keep `requirements_labeler.txt` as a generated compatibility export for `Launch_Labeler.bat`.
- Put every full sentence in long Markdown files on its own physical line.
- Do not use em dashes.
- Do not commit or push the implementation.

---

### Task 1: Track the empty repository skeleton safely

**Files:**

- Create: `tests/test_repository_reproducibility.py`
- Create: `.gitattributes`
- Create: `dataset/train/litoria_aurea/.gitkeep`
- Create: `dataset/val/litoria_aurea/.gitkeep`
- Create: `dataset/test/litoria_aurea/.gitkeep`
- Modify: `.gitignore`
- Remove: `dataset/train/bell_frog/.gitkeep`
- Remove: `dataset/val/bell_frog/.gitkeep`
- Remove: `dataset/test/bell_frog/.gitkeep`
- Test: `tests/test_repository_reproducibility.py`

**Interfaces:**

- Consumes: The existing directory skeleton and Git executable.
- Produces: `git_path_is_ignored(relative_path: str) -> bool` and a canonical `PLACEHOLDERS` tuple reused by later repository tests.

- [x] **Step 1: Write the failing layout and ignore test**

```python
from __future__ import annotations

import subprocess
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
    "dataset/train/litoria_aurea/.gitkeep",
    "dataset/train/non_target/.gitkeep",
    "dataset/val/litoria_aurea/.gitkeep",
    "dataset/val/non_target/.gitkeep",
    "dataset/test/litoria_aurea/.gitkeep",
    "dataset/test/non_target/.gitkeep",
    "models/.gitkeep",
    "results/analytics/.gitkeep",
)

GENERATED_PATHS = (
    "raw/external/example.WAV",
    "processed/external/chunks/example.wav",
    "processed/external/spectrograms/example.png",
    "labeled/litoria_aurea/example.png",
    "dataset/train/litoria_aurea/example.png",
    "models/example.pkl",
    "results/analytics/example.csv",
)


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

        for split in ("train", "val", "test"):
            self.assertFalse((REPO_ROOT / "dataset" / split / "bell_frog" / ".gitkeep").exists())


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run the test and verify the intended failure**

Run: `.venv\Scripts\python.exe -m unittest tests.test_repository_reproducibility.RepositoryLayoutTests -v`

Expected: FAIL because current placeholders are ignored and the canonical `litoria_aurea` dataset placeholders do not exist.

- [x] **Step 3: Replace the ignore rules and class placeholders**

Use recursive content rules for `raw/`, `processed/`, `labeled/`, `dataset/`, `models/`, and `results/`.
Re-include directories under those roots, then re-include `.gitkeep` files.
Add the three canonical positive-class placeholders before removing the three obsolete `bell_frog` placeholders.

```gitignore
# Data and generated artifacts
raw/**
processed/**
labeled/**
dataset/**
models/**
results/**

# Keep directory structure while ignoring generated contents
!raw/**/
!processed/**/
!labeled/**/
!dataset/**/
!models/**/
!results/**/
!**/.gitkeep
```

- [x] **Step 4: Run the layout test and verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_repository_reproducibility.RepositoryLayoutTests -v`

Expected: PASS with one test and no warnings.

### Task 2: Establish authoritative dependency metadata and locking

**Files:**

- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `uv.lock` with `uv lock`
- Modify: `requirements_labeler.txt` with `uv export`
- Modify: `Launch_Labeler.bat`
- Modify: `tests/test_repository_reproducibility.py`
- Test: `tests/test_repository_reproducibility.py`

**Interfaces:**

- Consumes: uv 0.11, the approved Python 3.13 policy, and the existing eight direct dependencies.
- Produces: A frozen uv environment and an exact pip-compatible runtime export consumed by `Launch_Labeler.bat`.

- [x] **Step 1: Add the failing dependency-contract test**

Add these imports and constants to `tests/test_repository_reproducibility.py`:

```python
import re
import tomllib

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
```

Add this test class:

```python
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
```

- [x] **Step 2: Run the dependency test and verify the intended failure**

Run: `.venv\Scripts\python.exe -m unittest tests.test_repository_reproducibility.DependencyMetadataTests -v`

Expected: FAIL because `pyproject.toml`, `.python-version`, and `uv.lock` do not exist.

- [x] **Step 3: Add project metadata and the Python version policy**

Create `.python-version` containing `3.13` followed by a newline.

Create `pyproject.toml` with:

```toml
[project]
name = "frog-classifier"
version = "0.1.0"
description = "Tools for preparing, labeling, training, and analyzing frog-call recordings."
requires-python = ">=3.13,<3.14"
dependencies = [
    "librosa>=0.11,<0.12",
    "matplotlib>=3.10,<3.11",
    "numpy>=2.3,<2.4",
    "pillow>=12.1,<12.2",
    "scikit-learn>=1.8,<1.9",
    "simpleaudio==1.0.4",
    "streamlit>=1.54,<1.55",
    "tqdm>=4.67,<4.68",
]

[tool.uv]
package = false
link-mode = "copy"
```

- [x] **Step 4: Generate the lock and compatibility export**

Run: `uv lock --python .venv\Scripts\python.exe`

Expected: `uv.lock` is created with a successful resolution for Python 3.13.

Run: `uv export --frozen --no-dev --no-hashes --format requirements-txt --output-file requirements_labeler.txt`

Expected: `requirements_labeler.txt` is replaced by a generated, exactly pinned runtime dependency graph.

- [x] **Step 5: Align the launcher with the generated lock export**

Keep the existing venv and pip bootstrap behavior.
Update comments and console messages to state that the launcher installs the locked runtime dependencies from the generated compatibility export.
Do not add a global uv prerequisite to the double-click path.

- [x] **Step 6: Run dependency verification**

Run: `.venv\Scripts\python.exe -m unittest tests.test_repository_reproducibility.DependencyMetadataTests -v`

Expected: PASS with one test and no warnings.

Run: `uv lock --check`

Expected: Exit code 0 and no lock changes.

Record the SHA-256 hash of `requirements_labeler.txt`, rerun the exact `uv export` command from Step 4, and record the new hash.

Expected: The hashes match, proving that the generated export is deterministic and current.

### Task 3: Align operational documentation and publish the roadmap

**Files:**

- Create: `roadmap.md`
- Modify: `PROCESS.md`
- Modify: `docs/system design.md`
- Modify: `scripts/slice_audio.py`
- Modify: `scripts/label_spectrograms.py`
- Modify: `scripts/build_manifest.py`
- Modify: `scripts/train_baseline.py`
- Modify: `tests/test_repository_reproducibility.py`
- Test: `tests/test_repository_reproducibility.py`

**Interfaces:**

- Consumes: The canonical repository layout and dependency commands from Tasks 1 and 2.
- Produces: Current operator guidance and a phase-based roadmap whose first phase can be marked complete after final verification.

- [x] **Step 1: Add the failing operational-documentation tests**

Add these constants:

```python
OPERATIONAL_TEXT_FILES = (
    "PROCESS.md",
    "docs/system design.md",
    "scripts/build_manifest.py",
    "scripts/label_spectrograms.py",
    "scripts/slice_audio.py",
    "scripts/train_baseline.py",
)

STALE_LAYOUT_TOKENS = ("Data/", "Data\\", "src/", "src\\")
```

Add this class:

```python
class OperationalDocumentationTests(unittest.TestCase):
    def test_operational_text_uses_the_canonical_layout(self) -> None:
        for relative_path in OPERATIONAL_TEXT_FILES:
            text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
            for stale_token in STALE_LAYOUT_TOKENS:
                with self.subTest(file=relative_path, token=stale_token):
                    self.assertNotIn(stale_token, text)

    def test_roadmap_covers_every_approved_phase(self) -> None:
        roadmap = (REPO_ROOT / "roadmap.md").read_text(encoding="utf-8")
        for phase in range(1, 8):
            with self.subTest(phase=phase):
                self.assertIn(f"## Phase {phase}:", roadmap)
        self.assertIn("Status: Next", roadmap)
        self.assertIn("Status: Planned", roadmap)
```

- [x] **Step 2: Run the documentation tests and verify the intended failures**

Run: `.venv\Scripts\python.exe -m unittest tests.test_repository_reproducibility.OperationalDocumentationTests -v`

Expected: FAIL because the old layout remains in operational text and `roadmap.md` does not exist.

- [x] **Step 3: Rewrite the process and system-design documentation**

Update all commands and diagrams to use:

```text
raw/external
raw/ponds
processed/external/spectrograms
processed/external/chunks
processed/ponds/spectrograms
processed/ponds/chunks
labeled/litoria_aurea
labeled/non_target
scripts/*.py
```

Document both setup paths:

```powershell
uv sync --frozen
```

```text
Double-click Launch_Labeler.bat
```

Document compatibility-export regeneration:

```powershell
uv lock
uv export --frozen --no-dev --no-hashes --format requirements-txt --output-file requirements_labeler.txt
```

- [x] **Step 4: Align script descriptions and help text**

Replace old `Data/` and `src/` examples with the canonical paths.
Use `non_target` for the negative class directory while retaining plain-language descriptions such as background noise where appropriate.
Do not change executable defaults or processing behavior.

- [x] **Step 5: Create the root roadmap**

Create `roadmap.md` with a checkpoint summary and seven phase sections.
For every phase, include status, purpose, concrete work, and acceptance criteria.
Mark Phase 1 as `Complete` only in the final verification update.
Mark Phase 2 as `Next` and the remaining phases as `Planned`.
Include the verified current counts of 618 raw recordings, 37,019 queued spectrograms, 13 positive labels, 48 negative labels, zero saved models, and zero result artifacts.

- [x] **Step 6: Run the documentation tests and verify they pass**

Run: `.venv\Scripts\python.exe -m unittest tests.test_repository_reproducibility.OperationalDocumentationTests -v`

Expected: PASS with two tests and no warnings.

### Task 4: Run the complete verification gate

**Files:**

- Modify: `roadmap.md` only if Phase 1 status requires correction based on verification.
- Verify: Every file changed by Tasks 1 through 3.

**Interfaces:**

- Consumes: The complete implementation.
- Produces: Fresh evidence that Phase 1 is complete or an exact list of remaining failures.

- [x] **Step 1: Run all repository tests**

Run: `.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: All tests pass with no warnings.

- [x] **Step 2: Verify dependency reproducibility**

Run: `uv lock --check`

Expected: Exit code 0.

Run: `uv sync --frozen`

Expected: Exit code 0 with `.venv` synchronized to the exact lock.

Run: `.venv\Scripts\python.exe -m pip check`

Expected: `No broken requirements found.`

- [x] **Step 3: Verify Python sources**

Run a Python 3.13 compilation pass over every `scripts/*.py` source without writing bytecode.

Expected: Five source files compile successfully.

- [x] **Step 4: Verify repository structure and data safety**

Run: `git check-ignore -v --no-index raw/external/example.WAV processed/external/spectrograms/example.png models/example.pkl results/analytics/example.csv`

Expected: Every representative generated path is ignored.

Run: `git check-ignore --no-index raw/external/.gitkeep dataset/train/litoria_aurea/.gitkeep models/.gitkeep`

Expected: Exit code 1 because placeholders are not ignored.

Recount local raw recordings, queued spectrograms, labeled images, model files, and result files.
Expected: The implementation did not alter any of those artifact counts.

- [x] **Step 5: Review the final diff and roadmap status**

Run: `git diff --check`

Expected: No whitespace errors.

Run: `git status --short`

Expected: Only the approved configuration, documentation, tests, lock files, and placeholders are changed or untracked.

Run: `git diff --stat` and inspect every changed file.
Expected: No generated data, audio, spectrogram, labels, model outputs, or result outputs are present in the diff.

If all checks pass, retain `Status: Complete` for Phase 1 in `roadmap.md`.
If any check fails, set Phase 1 to `Status: In progress` and record the exact remaining blocker before handoff.
