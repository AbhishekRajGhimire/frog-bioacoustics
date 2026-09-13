from __future__ import annotations

"""
sync_labeled_images.py

After spectrograms are regenerated, every labeled example has a fresh copy
in the queue with the same file name. This command moves each fresh copy
over its labeled counterpart so the label keeps its identity and the queue
no longer contains a labeled example.

All checks run before any file is touched. Any issue aborts the command
with every issue listed and nothing changed.
"""

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from frog_classifier.data import (
    ExampleNameError,
    load_preprocessing_config,
    parse_example_filename,
)


@dataclass(frozen=True)
class SyncPlan:
    replacements: tuple[tuple[Path, Path], ...]
    issues: tuple[str, ...]


def plan_sync(
    labeled_root: Path,
    spectrogram_root: Path,
    *,
    chunk_seconds: int,
) -> SyncPlan:
    labeled_root = labeled_root.resolve()
    spectrogram_root = spectrogram_root.resolve()
    if _is_within(labeled_root, spectrogram_root) or _is_within(spectrogram_root, labeled_root):
        return SyncPlan((), (
            f"[nested_roots] {labeled_root} and {spectrogram_root} must not contain each other",
        ))
    if not labeled_root.is_dir():
        return SyncPlan((), (f"[missing_root] labeled root is not a directory: {labeled_root}",))
    if not spectrogram_root.is_dir():
        return SyncPlan((), (
            f"[missing_root] spectrogram root is not a directory: {spectrogram_root}",
        ))

    fresh_by_name: dict[str, list[Path]] = defaultdict(list)
    for path in spectrogram_root.rglob("*.png"):
        if path.is_file():
            fresh_by_name[path.name].append(path)

    labeled_files = [path for path in labeled_root.rglob("*.png") if path.is_file()]
    labeled_by_name: dict[str, list[Path]] = defaultdict(list)
    for path in labeled_files:
        labeled_by_name[path.name].append(path)

    replacements: list[tuple[Path, Path]] = []
    issues: list[str] = []
    for labeled in sorted(labeled_files):
        try:
            parse_example_filename(labeled, chunk_seconds=chunk_seconds)
        except ExampleNameError as error:
            issues.append(f"[invalid_example_filename] {labeled}: {error}")
            continue
        duplicates = labeled_by_name[labeled.name]
        if len(duplicates) > 1:
            issues.append(
                f"[duplicate_labeled_name] {labeled}: {len(duplicates)} labeled images "
                "share this name"
            )
            continue
        candidates = sorted(fresh_by_name.get(labeled.name, []))
        if not candidates:
            issues.append(
                f"[missing_counterpart] {labeled}: no image named {labeled.name} "
                f"under {spectrogram_root}"
            )
        elif len(candidates) > 1:
            issues.append(
                f"[ambiguous_counterpart] {labeled}: {len(candidates)} images named "
                f"{labeled.name} under {spectrogram_root}"
            )
        else:
            replacements.append((candidates[0], labeled))
    return SyncPlan(tuple(replacements), tuple(issues))


def apply_sync(plan: SyncPlan) -> int:
    """Move each fresh image over its labeled counterpart. Returns the count moved."""
    for fresh, labeled in plan.replacements:
        os.replace(fresh, labeled)
    return len(plan.replacements)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replace labeled images with their freshly regenerated counterparts.",
    )
    parser.add_argument("--labeled-root", type=Path, default=Path("labeled"))
    parser.add_argument(
        "--spectrogram-root",
        type=Path,
        default=Path("processed/external/spectrograms"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    args = build_parser().parse_args(argv)
    config = load_preprocessing_config(repo_root / "config" / "preprocessing.toml")
    plan = plan_sync(
        _resolve(repo_root, args.labeled_root),
        _resolve(repo_root, args.spectrogram_root),
        chunk_seconds=config.audio.chunk_seconds,
    )
    if plan.issues:
        print("Labeled image sync failed; no files were changed:", file=sys.stderr)
        for issue in plan.issues:
            print(issue, file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"Dry run: {len(plan.replacements)} labeled image(s) would be replaced")
        return 0
    replaced = apply_sync(plan)
    print(f"Replaced {replaced} labeled image(s) from {args.spectrogram_root}")
    return 0


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
