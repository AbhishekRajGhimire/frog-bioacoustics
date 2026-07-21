from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from sklearn.model_selection import StratifiedGroupKFold

from .manifest import LabeledExample
from .validation import ManifestIssue, ManifestValidationError


@dataclass(frozen=True)
class SplitPlan:
    folds: int
    seed: int
    test_fold: int
    val_fold: int
    fold_by_recording: Mapping[str, int]
    split_by_recording: Mapping[str, str]


def create_split_plan(
    examples: Sequence[LabeledExample],
    *,
    folds: int = 5,
    seed: int = 1337,
    test_fold: int = 0,
    val_fold: int = 1,
) -> SplitPlan:
    if folds < 3:
        raise ManifestValidationError((ManifestIssue(
            "invalid_fold_count",
            "at least three folds are required",
            str(folds),
        ),))
    if (
        test_fold == val_fold
        or not 0 <= test_fold < folds
        or not 0 <= val_fold < folds
    ):
        raise ManifestValidationError((ManifestIssue(
            "invalid_fold_selection",
            "test and validation folds must be distinct and in range",
        ),))

    ordered = sorted(
        examples,
        key=lambda row: (row.recording_id, row.start_s, row.image_path),
    )
    groups_by_label: dict[int, set[str]] = defaultdict(set)
    for example in ordered:
        groups_by_label[example.label].add(example.recording_id)
    issues = tuple(
        ManifestIssue(
            "insufficient_class_groups",
            f"label {label} has {len(groups)} groups but {folds} are required",
            str(label),
        )
        for label, groups in sorted(groups_by_label.items())
        if len(groups) < folds
    )
    if issues:
        raise ManifestValidationError(issues)

    labels = [example.label for example in ordered]
    groups = [example.recording_id for example in ordered]
    assignments = [-1] * len(ordered)
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    for fold, (_, held_out) in enumerate(
        splitter.split([0] * len(ordered), labels, groups)
    ):
        for index in held_out:
            assignments[int(index)] = fold

    expected_labels = set(groups_by_label)
    fold_issues = []
    for fold in range(folds):
        observed = {
            labels[index]
            for index, assigned in enumerate(assignments)
            if assigned == fold
        }
        if observed != expected_labels:
            fold_issues.append(ManifestIssue(
                "fold_missing_class",
                f"fold has labels {sorted(observed)}",
                str(fold),
            ))
    if fold_issues:
        raise ManifestValidationError(fold_issues)

    fold_by_recording: dict[str, int] = {}
    for example, fold in zip(ordered, assignments, strict=True):
        previous = fold_by_recording.setdefault(example.recording_id, fold)
        if previous != fold:
            raise AssertionError("StratifiedGroupKFold split one recording group")
    split_by_recording = {
        recording_id: (
            "test"
            if fold == test_fold
            else "val"
            if fold == val_fold
            else "train"
        )
        for recording_id, fold in fold_by_recording.items()
    }
    return SplitPlan(
        folds=folds,
        seed=seed,
        test_fold=test_fold,
        val_fold=val_fold,
        fold_by_recording=MappingProxyType(fold_by_recording),
        split_by_recording=MappingProxyType(split_by_recording),
    )
