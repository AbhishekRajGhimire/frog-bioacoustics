from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    from .manifest import ManifestRow


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SPLITS = ("train", "val", "test")


@dataclass(frozen=True, order=True)
class ManifestIssue:
    code: str
    message: str
    subject: str = ""


class ManifestValidationError(ValueError):
    def __init__(self, issues: Iterable[ManifestIssue]):
        self.issues = tuple(sorted(issues))
        super().__init__(
            "\n".join(
                f"[{issue.code}] {issue.subject}: {issue.message}"
                for issue in self.issues
            )
        )


def validate_manifest_rows(
    rows: Sequence[ManifestRow],
    repo_root: Path,
    classes: Mapping[str, int],
    config_sha256: str,
    require_files: bool = True,
) -> None:
    issues: list[ManifestIssue] = []
    expected_labels = set(classes.values())
    expected_digest = config_sha256.lower()
    identities: dict[tuple[str, int], list[str]] = defaultdict(list)
    normalized_paths: dict[str, list[str]] = defaultdict(list)
    folds_by_recording: dict[str, set[int]] = defaultdict(set)
    splits_by_recording: dict[str, set[str]] = defaultdict(set)
    splits_by_fold: dict[int, set[str]] = defaultdict(set)
    labels_by_fold: dict[int, set[int]] = defaultdict(set)
    labels_by_split: dict[str, set[int]] = defaultdict(set)
    observed_folds: set[int] = set()
    repo_root = repo_root.resolve()

    for row in rows:
        if type(row.manifest_version) is not int or row.manifest_version != 1:
            issues.append(ManifestIssue(
                "unsupported_manifest_version",
                "manifest version must be 1",
                row.example_id,
            ))

        expected_label = classes.get(row.label_name)
        if expected_label is None or row.label != expected_label:
            issues.append(ManifestIssue(
                "label_mapping_mismatch",
                "label and label name do not match the configured classes",
                row.example_id,
            ))

        valid_split = row.split in _SPLITS
        if not valid_split:
            issues.append(ManifestIssue(
                "invalid_split",
                "split must be train, val, or test",
                row.example_id,
            ))

        valid_fold = type(row.fold) is int and row.fold >= 0
        if not valid_fold:
            issues.append(ManifestIssue(
                "invalid_fold",
                "fold must be a nonnegative integer",
                row.example_id,
            ))

        if (
            _SHA256_PATTERN.fullmatch(row.preprocessing_config_sha256) is None
            or row.preprocessing_config_sha256 != expected_digest
        ):
            issues.append(ManifestIssue(
                "invalid_preprocessing_config_sha256",
                "preprocessing digest must be lowercase SHA-256 matching the configuration",
                row.example_id,
            ))

        identities[(row.recording_id, row.start_s)].append(row.example_id)
        normalized_paths[row.image_path.casefold()].append(row.image_path)
        if valid_fold:
            observed_folds.add(row.fold)
            folds_by_recording[row.recording_id].add(row.fold)
            labels_by_fold[row.fold].add(row.label)
        if valid_split:
            splits_by_recording[row.recording_id].add(row.split)
            labels_by_split[row.split].add(row.label)
        if valid_fold and valid_split:
            splits_by_fold[row.fold].add(row.split)

        image_path = Path(row.image_path)
        if image_path.is_absolute():
            issues.append(ManifestIssue(
                "invalid_image_path",
                "image path must be repository-relative",
                row.image_path,
            ))
            continue
        resolved_path = (repo_root / image_path).resolve()
        try:
            resolved_path.relative_to(repo_root)
        except ValueError:
            issues.append(ManifestIssue(
                "invalid_image_path",
                "image path must remain inside the repository root",
                row.image_path,
            ))
            continue
        if require_files and not resolved_path.is_file():
            issues.append(ManifestIssue(
                "missing_image_file",
                "manifest image file does not exist",
                row.image_path,
            ))

    for (recording_id, start_s), example_ids in sorted(identities.items()):
        if len(example_ids) > 1:
            issues.append(ManifestIssue(
                "duplicate_example_identity",
                "multiple rows share the same recording and start time",
                f"{recording_id}_start{start_s}s",
            ))

    for normalized_path, image_paths in sorted(normalized_paths.items()):
        if len(image_paths) > 1:
            issues.append(ManifestIssue(
                "duplicate_image_path",
                "multiple manifest paths differ only by case",
                normalized_path,
            ))

    for recording_id, recording_folds in sorted(folds_by_recording.items()):
        if len(recording_folds) > 1:
            issues.append(ManifestIssue(
                "recording_fold_leakage",
                f"recording appears in folds {sorted(recording_folds)}",
                recording_id,
            ))

    for recording_id, recording_splits in sorted(splits_by_recording.items()):
        if len(recording_splits) > 1:
            issues.append(ManifestIssue(
                "recording_split_leakage",
                f"recording appears in splits {sorted(recording_splits)}",
                recording_id,
            ))

    expected_folds = set(range(max(observed_folds) + 1)) if observed_folds else set()
    if observed_folds != expected_folds:
        missing_folds = sorted(expected_folds - observed_folds)
        issues.append(ManifestIssue(
            "non_contiguous_folds",
            f"fold numbers must be contiguous from zero; missing {missing_folds}",
        ))

    for fold, fold_splits in sorted(splits_by_fold.items()):
        if len(fold_splits) > 1:
            issues.append(ManifestIssue(
                "fold_split_inconsistency",
                f"fold appears in splits {sorted(fold_splits)}",
                str(fold),
            ))

    folds_by_split = {
        split: {
            fold
            for fold, fold_splits in splits_by_fold.items()
            if split in fold_splits
        }
        for split in _SPLITS
    }
    if len(folds_by_split["test"]) != 1:
        issues.append(ManifestIssue(
            "invalid_test_fold_count",
            "test split must contain exactly one fold",
        ))
    if len(folds_by_split["val"]) != 1:
        issues.append(ManifestIssue(
            "invalid_validation_fold_count",
            "validation split must contain exactly one fold",
        ))
    if not folds_by_split["train"]:
        issues.append(ManifestIssue(
            "missing_training_fold",
            "training split must contain at least one fold",
        ))

    for fold in sorted(observed_folds):
        observed_labels = labels_by_fold[fold]
        if observed_labels != expected_labels:
            issues.append(ManifestIssue(
                "fold_missing_class",
                f"fold has labels {sorted(observed_labels)}",
                str(fold),
            ))

    for split in _SPLITS:
        observed_labels = labels_by_split[split]
        if observed_labels != expected_labels:
            issues.append(ManifestIssue(
                "split_missing_class",
                f"split has labels {sorted(observed_labels)}",
                split,
            ))

    if issues:
        raise ManifestValidationError(issues)
