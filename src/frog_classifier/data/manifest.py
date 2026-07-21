from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from .config import PreprocessingConfig
from .naming import ExampleNameError, ExampleKey, parse_example_filename
from .validation import ManifestIssue, ManifestValidationError

if TYPE_CHECKING:
    from .splitting import SplitPlan


@dataclass(frozen=True)
class LabeledExample:
    example_id: str
    image_path: str
    label: int
    label_name: str
    recording_id: str
    start_s: int


@dataclass(frozen=True)
class ManifestRow:
    manifest_version: int
    example_id: str
    image_path: str
    label: int
    label_name: str
    recording_id: str
    start_s: int
    fold: int
    split: str
    preprocessing_config_sha256: str


def build_manifest_rows(
    examples: Sequence[LabeledExample],
    plan: SplitPlan,
    config_sha256: str,
) -> tuple[ManifestRow, ...]:
    digest = config_sha256.lower()
    return tuple(
        ManifestRow(
            manifest_version=1,
            example_id=example.example_id,
            image_path=example.image_path,
            label=example.label,
            label_name=example.label_name,
            recording_id=example.recording_id,
            start_s=example.start_s,
            fold=plan.fold_by_recording[example.recording_id],
            split=plan.split_by_recording[example.recording_id],
            preprocessing_config_sha256=digest,
        )
        for example in examples
    )


def discover_labeled_examples(
    label_root: Path,
    repo_root: Path,
    config: PreprocessingConfig,
) -> tuple[LabeledExample, ...]:
    label_root = label_root.resolve()
    repo_root = repo_root.resolve()
    issues: list[ManifestIssue] = []
    expected_labels = set(config.classes)

    for label_name in sorted(expected_labels):
        if not (label_root / label_name).is_dir():
            issues.append(
                ManifestIssue(
                    "missing_label_directory",
                    "required label directory is missing",
                    label_name,
                )
            )

    image_paths = sorted(
        (
            path
            for path in label_root.rglob("*")
            if path.suffix.casefold() == ".png"
        ),
        key=lambda path: path.as_posix(),
    ) if label_root.is_dir() else []

    candidates: list[tuple[Path, str, str, ExampleKey]] = []
    normalized_paths: dict[str, list[str]] = defaultdict(list)
    identities: dict[tuple[str, int], list[str]] = defaultdict(list)

    for path in image_paths:
        if not path.is_file():
            issues.append(
                ManifestIssue(
                    "not_regular_file",
                    "PNG path is not a regular file",
                    path.as_posix(),
                )
            )
            continue

        relative_to_labels = path.relative_to(label_root)
        label_name = relative_to_labels.parts[0]
        try:
            image_path = path.relative_to(repo_root).as_posix()
        except ValueError:
            issues.append(
                ManifestIssue(
                    "image_path_outside_repository",
                    "image path must be inside the repository root",
                    path.as_posix(),
                )
            )
            continue

        normalized_paths[image_path.casefold()].append(image_path)
        if label_name not in expected_labels:
            issues.append(
                ManifestIssue(
                    "unknown_label_directory",
                    "label directory is not in the preprocessing configuration",
                    label_name,
                )
            )

        try:
            key = parse_example_filename(
                path,
                chunk_seconds=config.audio.chunk_seconds,
            )
        except ExampleNameError as error:
            issues.append(
                ManifestIssue("invalid_example_filename", str(error), image_path)
            )
            continue

        identities[(key.recording_id, key.start_s)].append(image_path)
        candidates.append((path, image_path, label_name, key))

    for normalized_path, paths in sorted(normalized_paths.items()):
        if len(paths) > 1:
            issues.append(
                ManifestIssue(
                    "duplicate_image_path",
                    "multiple image paths differ only by case",
                    normalized_path,
                )
            )

    for (recording_id, start_s), paths in sorted(identities.items()):
        if len(paths) > 1:
            issues.append(
                ManifestIssue(
                    "duplicate_example_identity",
                    "multiple images share the same recording and start time",
                    f"{recording_id}_start{start_s}s",
                )
            )

    if issues:
        raise ManifestValidationError(issues)

    examples = tuple(
        LabeledExample(
            example_id=key.example_id,
            image_path=image_path,
            label=config.classes[label_name],
            label_name=label_name,
            recording_id=key.recording_id,
            start_s=key.start_s,
        )
        for _, image_path, label_name, key in candidates
    )
    return tuple(sorted(examples, key=lambda example: example.image_path))
