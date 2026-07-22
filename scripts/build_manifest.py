from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from frog_classifier.data import (
    ConfigError,
    ManifestIssue,
    ManifestValidationError,
    build_data_quality_report,
    build_manifest_rows,
    create_split_plan,
    discover_labeled_examples,
    load_preprocessing_config,
    serialize_manifest,
    serialize_report_json,
    serialize_report_markdown,
    validate_manifest_rows,
    write_output_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a validated, leakage-safe labeled-image manifest.",
    )
    parser.add_argument("--training-root", type=Path, default=Path("labeled"))
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("labeled/manifest.csv"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("results/data_quality"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/preprocessing.toml"),
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--val-fold", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    args = build_parser().parse_args(argv)
    training_root = _resolve(repo_root, args.training_root)
    out_csv = _resolve(repo_root, args.out_csv)
    report_dir = _resolve(repo_root, args.report_dir)
    config_path = _resolve(repo_root, args.config)
    report_json = (report_dir / "manifest-report.json").resolve()
    report_markdown = (report_dir / "manifest-report.md").resolve()

    try:
        config = load_preprocessing_config(config_path)
        examples = discover_labeled_examples(training_root, repo_root, config)
        _validate_output_paths(
            out_csv=out_csv,
            report_paths=(report_json, report_markdown),
            repo_root=repo_root,
            training_root=training_root,
            config_path=config_path,
            discovered_image_paths=tuple(
                (repo_root / example.image_path).resolve()
                for example in examples
            ),
        )
        plan = create_split_plan(
            examples,
            folds=args.folds,
            seed=args.seed,
            test_fold=args.test_fold,
            val_fold=args.val_fold,
        )
        rows = build_manifest_rows(examples, plan, config.sha256)
        validate_manifest_rows(
            rows,
            repo_root,
            config.classes,
            config.sha256,
            config.audio.chunk_seconds,
        )
        manifest_payload = serialize_manifest(rows)
        report = build_data_quality_report(
            rows,
            plan,
            config,
            manifest_payload,
            repo_root,
        )
        write_output_bundle({
            out_csv: manifest_payload,
            report_json: serialize_report_json(report),
            report_markdown: serialize_report_markdown(report),
        })
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except ManifestValidationError as error:
        print("Manifest validation failed:", file=sys.stderr)
        print(str(error), file=sys.stderr)
        return 2

    print(f"Manifest: {out_csv}")
    print(f"JSON report: {report_json}")
    print(f"Markdown report: {report_markdown}")
    print(
        f"Examples: {report.total.examples} | "
        f"recording groups: {report.total.recording_groups}"
    )
    for class_name, summary in report.by_class.items():
        print(
            f"{class_name}: {summary.examples} examples | "
            f"{summary.recording_groups} recording groups"
        )
    return 0


def _resolve(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _validate_output_paths(
    *,
    out_csv: Path,
    report_paths: Sequence[Path],
    repo_root: Path,
    training_root: Path,
    config_path: Path,
    discovered_image_paths: Sequence[Path],
) -> None:
    paths = (out_csv, *report_paths)
    issues: list[ManifestIssue] = []
    if out_csv.suffix.casefold() != ".csv":
        issues.append(ManifestIssue(
            "invalid_output_suffix",
            "manifest destination must have a CSV suffix",
            str(out_csv),
        ))

    collisions = sorted(
        {path for path in paths if paths.count(path) > 1},
        key=str,
    )
    issues.extend(
        ManifestIssue(
            "output_path_collision",
            "manifest CSV and report destinations must be pairwise distinct",
            str(path),
        )
        for path in collisions
    )

    repo_root = repo_root.resolve()
    training_root = training_root.resolve()
    default_training_root = (repo_root / "labeled").resolve()
    default_manifest = (default_training_root / "manifest.csv").resolve()
    protected_files = {
        config_path.resolve(),
        *(path.resolve() for path in discovered_image_paths),
    }
    protected_roots = (
        (repo_root / "raw").resolve(),
        (repo_root / "processed").resolve(),
        (repo_root / "config").resolve(),
        training_root,
    )
    for path in paths:
        resolved_path = path.resolve()
        allowed_default_manifest = (
            resolved_path == out_csv.resolve() == default_manifest
            and training_root == default_training_root
        )
        protected = resolved_path in protected_files or any(
            _is_within(resolved_path, root)
            for root in protected_roots
        )
        if protected and not allowed_default_manifest:
            issues.append(
                ManifestIssue(
                    "protected_output_path",
                    (
                        "output destination must not replace configuration, raw, "
                        "processed, or labeled source data"
                    ),
                    str(resolved_path),
                )
            )

    if issues:
        raise ManifestValidationError(issues)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
