"""
slice_audio.py

Slice every recording beneath a raw root into fixed five-second windows and
write one Mel-spectrogram PNG per complete window, preserving the folder
structure beneath the output root.

Every preprocessing value comes from config/preprocessing.toml through
frog_classifier.preprocessing. The command deliberately offers no overrides,
so stored images can never drift from the tracked contract.

Output naming: <stem>_start<N>s.png, where N is the window start in seconds.
Existing images with the same name are overwritten. Nothing is deleted.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from tqdm import tqdm

from frog_classifier.data import PreprocessingConfig, load_preprocessing_config
from frog_classifier.preprocessing import render_recording


SUPPORTED_EXTS = {".wav", ".mp3"}


def iter_audio_files_os_walk(raw_root: Path) -> Iterable[Path]:
    """Find every supported audio file beneath raw_root."""
    for dirpath, _, filenames in os.walk(raw_root):
        directory = Path(dirpath)
        for name in filenames:
            path = directory / name
            if path.suffix.lower() in SUPPORTED_EXTS:
                yield path


def pond_name_for_path(raw_root: Path, audio_path: Path) -> str:
    """The top-level folder beneath raw_root, used to group progress."""
    relative = audio_path.relative_to(raw_root)
    return relative.parts[0] if len(relative.parts) > 0 else "raw"


def process_file(
    config: PreprocessingConfig,
    raw_root: Path,
    out_root: Path,
    audio_path: Path,
) -> int:
    """Render one recording and write its chunk images. Returns the count written."""
    relative = audio_path.relative_to(raw_root)
    out_dir = out_root / relative.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for chunk in render_recording(audio_path, config):
        out_path = out_dir / f"{audio_path.stem}_start{chunk.start_s}s.png"
        out_path.write_bytes(chunk.png_bytes)
        written += 1
    return written


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Slice raw audio into fixed windows and save Mel-spectrogram PNGs "
            "rendered against each recording's noise floor."
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=repo_root / "raw" / "external",
        help='Input root containing nested audio files (default: "raw/external").',
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=repo_root / "processed" / "external" / "spectrograms",
        help='Output root for spectrogram PNGs (default: "processed/external/spectrograms").',
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="",
        help="Optional subfolder under out-root.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="Process only the first N audio files per pond (0 = no limit).",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, repo_root: Path | None = None) -> int:
    repo_root = repo_root if repo_root is not None else Path(__file__).resolve().parents[1]
    args = build_parser(repo_root).parse_args(argv)
    config = load_preprocessing_config(repo_root / "config" / "preprocessing.toml")

    raw_root = _resolve_path(repo_root, args.raw_root)
    out_root = _resolve_path(repo_root, args.out_root) / (args.out_subdir or "")
    labeled_root = repo_root / "labeled"
    if _is_within(out_root, labeled_root):
        print(f"out_root must not lie inside {labeled_root}: {out_root}", file=sys.stderr)
        return 2
    if _is_within(out_root, raw_root):
        print(f"out_root must not lie inside raw_root {raw_root}: {out_root}", file=sys.stderr)
        return 2
    if _is_within(raw_root, out_root):
        print(f"raw_root must not lie inside out_root {out_root}: {raw_root}", file=sys.stderr)
        return 2
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")

    pond_to_files: dict[str, list[Path]] = defaultdict(list)
    for path in iter_audio_files_os_walk(raw_root):
        pond_to_files[pond_name_for_path(raw_root, path)].append(path)

    if not pond_to_files:
        print(f"No audio files found under {raw_root} (supported: {sorted(SUPPORTED_EXTS)})")
        return 0

    total_written = 0
    for pond in sorted(pond_to_files):
        files = sorted(pond_to_files[pond])
        if args.limit_files > 0:
            files = files[: args.limit_files]
        bar = tqdm(files, desc=f"Processing pond {pond}", unit="file")
        for audio_path in bar:
            try:
                total_written += process_file(config, raw_root, out_root, audio_path)
            except Exception as error:
                # Keep the run going; one unreadable file must not lose the batch.
                bar.write(f"[ERROR] {audio_path}: {error}")

    print(f"Done. Wrote {total_written} spectrogram PNG(s) to {out_root}")
    return 0


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
