from __future__ import annotations

"""
verify_spectrograms.py

Prove that stored spectrogram images were rendered under the tracked
preprocessing contract by re-rendering their recordings from raw audio and
comparing bytes. Samples from both the labeled tree and the queue, or checks
everything with --all.
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from frog_classifier.data import (
    ExampleNameError,
    PreprocessingConfig,
    load_preprocessing_config,
    parse_example_filename,
)
from frog_classifier.preprocessing import render_recording


SUPPORTED_EXTS = {".wav", ".mp3"}


@dataclass(frozen=True)
class VerificationResult:
    image_path: Path
    status: str


def find_recording(raw_root: Path, stem: str) -> Path | None:
    """The single recording with this stem beneath raw_root, or None."""
    matches = sorted(
        path
        for path in raw_root.rglob(f"{stem}.*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
    )
    return matches[0] if len(matches) == 1 else None


def verify_images(
    image_paths: Iterable[Path],
    raw_root: Path,
    config: PreprocessingConfig,
) -> list[VerificationResult]:
    results: list[VerificationResult] = []
    by_stem: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for path in image_paths:
        try:
            key = parse_example_filename(path, chunk_seconds=config.audio.chunk_seconds)
        except ExampleNameError:
            results.append(VerificationResult(path, "INVALID_NAME"))
            continue
        by_stem[key.recording_id].append((path, key.start_s))

    for stem, items in sorted(by_stem.items()):
        recording = find_recording(raw_root, stem)
        if recording is None:
            results.extend(VerificationResult(path, "MISSING_RECORDING") for path, _ in items)
            continue
        rendered = {chunk.start_s: chunk.png_bytes for chunk in render_recording(recording, config)}
        for path, start_s in items:
            expected = rendered.get(start_s)
            if expected is None:
                status = "MISSING_CHUNK"
            elif path.read_bytes() == expected:
                status = "MATCH"
            else:
                status = "DIFFERENT"
            results.append(VerificationResult(path, status))
    return sorted(results, key=lambda result: str(result.image_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-render sampled spectrograms from raw audio and compare bytes.",
    )
    parser.add_argument("--raw-root", type=Path, default=Path("raw/external"))
    parser.add_argument(
        "--spectrogram-root",
        type=Path,
        default=Path("processed/external/spectrograms"),
    )
    parser.add_argument("--labeled-root", type=Path, default=Path("labeled"))
    parser.add_argument("--sample", type=int, default=24, help="Images per tree to check.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--all", action="store_true", help="Check every image in both trees.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    args = build_parser().parse_args(argv)
    config = load_preprocessing_config(repo_root / "config" / "preprocessing.toml")
    raw_root = _resolve(repo_root, args.raw_root)
    trees = (
        _resolve(repo_root, args.labeled_root),
        _resolve(repo_root, args.spectrogram_root),
    )

    selected: list[Path] = []
    rng = random.Random(args.seed)
    for tree in trees:
        images = sorted(path for path in tree.rglob("*.png") if path.is_file())
        if args.all:
            selected.extend(images)
        else:
            selected.extend(rng.sample(images, min(args.sample, len(images))))

    if not selected:
        print("No images found to verify", file=sys.stderr)
        return 2

    results = verify_images(selected, raw_root, config)
    for result in results:
        print(f"{result.status} {result.image_path}")
    counts = Counter(result.status for result in results)
    print(f"{len(results)} checked, {counts.get('MATCH', 0)} matched")
    for status in sorted(status for status in counts if status != "MATCH"):
        print(f"{counts[status]} {status}", file=sys.stderr)
    return 0 if counts.get("MATCH", 0) == len(results) else 2


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


if __name__ == "__main__":
    raise SystemExit(main())
