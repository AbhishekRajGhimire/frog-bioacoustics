from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .config import PreprocessingConfig
from .manifest import ManifestRow, serialize_manifest
from .splitting import SplitPlan
from .validation import (
    ManifestIssue,
    ManifestValidationError,
    validate_manifest_rows,
)


PROVENANCE_NOTE = (
    "The preprocessing checksum records the declared configuration but does "
    "not cryptographically prove how pre-existing PNG files were generated."
)


@dataclass(frozen=True)
class CountSummary:
    examples: int
    recording_groups: int


@dataclass(frozen=True)
class PartitionSummary:
    examples: int
    recording_groups: int
    examples_by_class: dict[str, int]
    recording_groups_by_class: dict[str, int]


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool


@dataclass(frozen=True)
class DataQualityReport:
    report_schema_version: int
    manifest_version: int
    seed: int
    folds: int
    test_fold: int
    val_fold: int
    preprocessing_config_path: str
    preprocessing_config_sha256: str
    manifest_sha256: str
    total: CountSummary
    by_class: dict[str, CountSummary]
    by_fold: dict[str, PartitionSummary]
    by_split: dict[str, PartitionSummary]
    validation_checks: tuple[ValidationCheck, ...]
    provenance_note: str


def build_data_quality_report(
    rows: Sequence[ManifestRow],
    plan: SplitPlan,
    config: PreprocessingConfig,
    manifest_payload: bytes,
    repo_root: Path,
    *,
    label_root: Path,
) -> DataQualityReport:
    validate_manifest_rows(
        rows,
        repo_root,
        config.classes,
        config.sha256,
        config.audio.chunk_seconds,
        label_root=label_root,
    )
    canonical_manifest_payload = serialize_manifest(rows)
    if manifest_payload != canonical_manifest_payload:
        raise ManifestValidationError((ManifestIssue(
            "manifest_payload_mismatch",
            (
                "manifest payload must equal the canonical serialization of "
                "the validated rows"
            ),
        ),))
    _validate_rows_match_plan(rows, plan)
    class_names = tuple(sorted(config.classes))
    by_class = {
        class_name: _count_summary(
            row for row in rows if row.label_name == class_name
        )
        for class_name in class_names
    }
    by_fold = {
        str(fold): _partition_summary(
            (row for row in rows if row.fold == fold),
            class_names,
        )
        for fold in range(plan.folds)
    }
    by_split = {
        split: _partition_summary(
            (row for row in rows if row.split == split),
            class_names,
        )
        for split in sorted(("train", "val", "test"))
    }
    return DataQualityReport(
        report_schema_version=1,
        manifest_version=1,
        seed=plan.seed,
        folds=plan.folds,
        test_fold=plan.test_fold,
        val_fold=plan.val_fold,
        preprocessing_config_path=_display_path(config.source_path, repo_root),
        preprocessing_config_sha256=config.sha256,
        manifest_sha256=hashlib.sha256(canonical_manifest_payload).hexdigest(),
        total=_count_summary(rows),
        by_class=by_class,
        by_fold=by_fold,
        by_split=by_split,
        validation_checks=(
            ValidationCheck("canonical_manifest_schema", True),
            ValidationCheck("preprocessing_configuration_matches", True),
            ValidationCheck("rows_match_split_plan", True),
            ValidationCheck("all_image_files_exist", True),
            ValidationCheck("recording_groups_are_isolated", True),
            ValidationCheck("every_fold_contains_all_classes", True),
            ValidationCheck("every_split_contains_all_classes", True),
        ),
        provenance_note=PROVENANCE_NOTE,
    )


def _validate_rows_match_plan(
    rows: Sequence[ManifestRow],
    plan: SplitPlan,
) -> None:
    issues: list[ManifestIssue] = []
    recording_ids = {row.recording_id for row in rows}
    fold_recording_ids = set(plan.fold_by_recording)
    split_recording_ids = set(plan.split_by_recording)
    if fold_recording_ids != recording_ids or split_recording_ids != recording_ids:
        issues.append(ManifestIssue(
            "split_plan_recordings_mismatch",
            "split plan recording IDs must exactly match the manifest rows",
        ))

    observed_folds = {row.fold for row in rows}
    expected_folds = set(range(plan.folds))
    if observed_folds != expected_folds:
        issues.append(ManifestIssue(
            "split_plan_fold_set_mismatch",
            (
                f"manifest folds {sorted(observed_folds)} do not match "
                f"planned folds {sorted(expected_folds)}"
            ),
        ))

    for row in rows:
        if plan.fold_by_recording.get(row.recording_id) != row.fold:
            issues.append(ManifestIssue(
                "split_plan_fold_mismatch",
                "row fold does not match the split plan",
                row.example_id,
            ))
        expected_split = (
            "test"
            if row.fold == plan.test_fold
            else "val"
            if row.fold == plan.val_fold
            else "train"
        )
        if (
            plan.split_by_recording.get(row.recording_id) != row.split
            or row.split != expected_split
        ):
            issues.append(ManifestIssue(
                "split_plan_split_mismatch",
                "row split does not match the split plan and selected folds",
                row.example_id,
            ))

    if issues:
        raise ManifestValidationError(issues)


def serialize_report_json(report: DataQualityReport) -> bytes:
    payload = json.dumps(
        asdict(report),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
    return payload.encode("utf-8")


def serialize_report_markdown(report: DataQualityReport) -> bytes:
    lines = [
        "# Manifest Data Quality Report",
        "",
        "## Reproducibility",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Report schema version | {report.report_schema_version} |",
        f"| Manifest version | {report.manifest_version} |",
        f"| Seed | {report.seed} |",
        f"| Folds | {report.folds} |",
        f"| Test fold | {report.test_fold} |",
        f"| Validation fold | {report.val_fold} |",
        f"| Preprocessing configuration | {report.preprocessing_config_path} |",
        f"| Preprocessing SHA-256 | `{report.preprocessing_config_sha256}` |",
        f"| Manifest SHA-256 | `{report.manifest_sha256}` |",
        "",
        "## Totals",
        "",
        "| Examples | Recording Groups |",
        "| ---: | ---: |",
        f"| {report.total.examples} | {report.total.recording_groups} |",
        "",
        "## Counts by Class",
        "",
        "| Class | Examples | Recording Groups |",
        "| --- | ---: | ---: |",
    ]
    for class_name, summary in report.by_class.items():
        lines.append(
            f"| {class_name} | {summary.examples} | {summary.recording_groups} |"
        )
    lines.extend(("", "## Counts by Fold", ""))
    lines.extend(_partition_table(report.by_fold, "Fold"))
    lines.extend(("", "## Counts by Split", ""))
    lines.extend(_partition_table(report.by_split, "Split"))
    lines.extend((
        "",
        "## Validation Checks",
        "",
        "| Check | Passed |",
        "| --- | --- |",
    ))
    for check in report.validation_checks:
        lines.append(f"| {check.name} | {'yes' if check.passed else 'no'} |")
    lines.extend((
        "",
        "## Provenance",
        "",
        report.provenance_note,
        "",
    ))
    return "\n".join(lines).encode("utf-8")


def write_output_bundle(outputs: Mapping[Path, bytes]) -> None:
    temporary_paths: dict[Path, Path] = {}
    try:
        for destination, payload in sorted(
            outputs.items(),
            key=lambda item: str(item[0]),
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary_file:
                temporary_paths[destination] = Path(temporary_file.name)
                temporary_file.write(payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
    except BaseException:
        _remove_temporary_files(temporary_paths.values())
        raise

    try:
        for destination, temporary_path in temporary_paths.items():
            temporary_path.replace(destination)
    finally:
        _remove_temporary_files(temporary_paths.values())


def _count_summary(rows: Iterable[ManifestRow]) -> CountSummary:
    materialized = tuple(rows)
    return CountSummary(
        examples=len(materialized),
        recording_groups=len({row.recording_id for row in materialized}),
    )


def _partition_summary(
    rows: Iterable[ManifestRow],
    class_names: tuple[str, ...],
) -> PartitionSummary:
    materialized = tuple(rows)
    return PartitionSummary(
        examples=len(materialized),
        recording_groups=len({row.recording_id for row in materialized}),
        examples_by_class={
            class_name: sum(
                row.label_name == class_name
                for row in materialized
            )
            for class_name in class_names
        },
        recording_groups_by_class={
            class_name: len({
                row.recording_id
                for row in materialized
                if row.label_name == class_name
            })
            for class_name in class_names
        },
    )


def _partition_table(
    partitions: Mapping[str, PartitionSummary],
    name_heading: str,
) -> list[str]:
    class_names = tuple(next(iter(partitions.values())).examples_by_class)
    headings = [name_heading, "Examples", "Recording Groups"]
    for class_name in class_names:
        headings.extend((
            f"{class_name} Examples",
            f"{class_name} Recording Groups",
        ))
    lines = [
        f"| {' | '.join(headings)} |",
        f"| {' | '.join(['---'] + ['---:'] * (len(headings) - 1))} |",
    ]
    for name, summary in partitions.items():
        values = [name, str(summary.examples), str(summary.recording_groups)]
        for class_name in class_names:
            values.extend((
                str(summary.examples_by_class[class_name]),
                str(summary.recording_groups_by_class[class_name]),
            ))
        lines.append(f"| {' | '.join(values)} |")
    return lines


def _display_path(path: Path, repo_root: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def _remove_temporary_files(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
