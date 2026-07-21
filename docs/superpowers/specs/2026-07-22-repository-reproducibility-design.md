# Repository Reproducibility Design

## Context

The repository has completed audio preprocessing and has begun manual labeling, but its tracked project structure does not yet reproduce the working local layout.
The April 2026 refactor moved scripts from `src/` to `scripts/` and data from `Data/` to top-level directories, while the process documentation and several script descriptions retained the old paths.
The current `.gitignore` rules ignore entire data directories before attempting to re-include `.gitkeep` files, so Git cannot track those placeholders.
The project currently declares only unpinned direct dependencies in `requirements_labeler.txt`.

## Goal

Make a fresh clone reproduce the intended repository structure, Python environment, documentation, and class vocabulary without changing audio processing, labeling, manifest, or model behavior.

## Non-goals

- Do not modify, move, or delete raw recordings, generated spectrograms, cached audio chunks, or labeled images.
- Do not change the manifest split algorithm or model-training behavior in this milestone.
- Do not build the inference engine, analytics pipeline, or classifier application in this milestone.
- Do not commit or push the resulting implementation unless the user requests it.

## Dependency management

`pyproject.toml` will become the authoritative declaration of project metadata, the supported Python release line, and direct runtime dependencies.
The project will support Python 3.13 through `requires-python = ">=3.13,<3.14"` and a root `.python-version` containing `3.13`.
Direct dependency constraints will use compatible minor release ranges so routine lock refreshes can receive patch releases while avoiding unreviewed feature-version upgrades.

`uv.lock` will provide the exact resolved dependency graph used by developers and automated checks.
The repository will be configured as a dependency-only application rather than an installable Python package because the current codebase consists of runnable scripts.
uv will use copy mode so synchronization behaves consistently when its cache and the project are on different Windows filesystems.
Developers will create or synchronize the environment with `uv sync --frozen`.

`requirements_labeler.txt` will remain for compatibility with the existing double-click Windows launcher.
It will be generated from `uv.lock` with production dependencies pinned to exact versions, rather than maintained manually.
The launcher will continue using `pip` inside `.venv`, which preserves the current non-technical setup path without requiring a global `uv` installation.
Documentation will identify `pyproject.toml` and `uv.lock` as authoritative and will include the exact regeneration command for the compatibility export.

## Repository layout and ignore behavior

The canonical class directory names will be `litoria_aurea` for the positive class and `non_target` for the negative class.
The unused `dataset/{train,val,test}/bell_frog/` placeholders will be replaced by `dataset/{train,val,test}/litoria_aurea/` placeholders.
Existing labeled data already uses the canonical names and will remain untouched.

The `.gitignore` data-directory rules will ignore all generated content while re-including directories needed to reach `.gitkeep` files.
Existing placeholders under `raw/`, `processed/`, `labeled/`, `dataset/`, `models/`, and `results/` will become trackable.
Representative audio, image, model, and result files will remain ignored.
`.gitattributes` will enforce LF line endings for text and mark project artifact formats as binary.

## Documentation alignment

A root `roadmap.md` will describe the current checkpoint and all planned phases:

1. Repository reproducibility.
2. Data integrity and trustworthy evaluation.
3. Purposeful expansion of human labels.
4. Reproducible model training and model persistence.
5. Batch inference and structured detections.
6. Activity analytics and the user-facing application.
7. Operational quality, monitoring, and maintenance.

Each phase will include its purpose, concrete work, acceptance criteria, and status.
The first phase will be marked complete only after its verification checks pass.

`PROCESS.md` and `docs/system design.md` will be updated to match the current top-level layout and `scripts/` commands.
References to the negative class will use `non_target` instead of `background` when naming a filesystem or manifest label.
User-facing script descriptions, examples, and argument help will use the same paths and class vocabulary.
Historical design intent may still describe background audio in prose, but operational names must remain canonical.

## Verification strategy

A focused `unittest` suite will be added for repository reproducibility.
The tests will use real Git ignore evaluation and real project files rather than mocks.

The suite will verify:

- A representative generated file in every data or artifact root is ignored.
- Every intended `.gitkeep` placeholder is not ignored and can therefore be tracked.
- Dataset split placeholders use `litoria_aurea` and `non_target`, with no tracked `bell_frog` placeholder.
- Operational documentation and script help text do not reference the removed `src/` or `Data/` layout.
- `pyproject.toml`, `.python-version`, `uv.lock`, and the generated compatibility requirements file agree on the supported Python release and runtime dependency set.

Tests for new repository behavior will be written first and observed failing before the configuration and documentation changes are applied.
After implementation, the complete verification pass will run the new unit tests, `uv lock --check`, a frozen environment synchronization check, Python compilation for all scripts, `pip check`, and a final Git diff review.

## Files

The implementation will create:

- `roadmap.md`
- `.gitattributes`
- `pyproject.toml`
- `.python-version`
- `uv.lock`
- `tests/test_repository_reproducibility.py`
- `dataset/train/litoria_aurea/.gitkeep`
- `dataset/val/litoria_aurea/.gitkeep`
- `dataset/test/litoria_aurea/.gitkeep`

The implementation will modify:

- `.gitignore`
- `requirements_labeler.txt` through `uv export`
- `Launch_Labeler.bat`
- `PROCESS.md`
- `docs/system design.md`
- Script module descriptions, examples, and argument help under `scripts/`

The obsolete `dataset/{train,val,test}/bell_frog/.gitkeep` placeholders will be removed after their canonical replacements exist.
No other data files or directories will be removed.

## Acceptance criteria

- A fresh clone contains the complete intended empty directory skeleton.
- Generated data remains excluded from Git.
- Developers can reproduce the exact environment with the frozen `uv` lock.
- The Windows launcher installs the exact exported runtime dependency set.
- All current commands and paths in operational documentation exist.
- Filesystem and manifest class names consistently use `litoria_aurea` and `non_target`.
- The reproducibility tests and all existing smoke checks pass without warnings caused by this milestone.
- The roadmap accurately marks only verified work as complete.
