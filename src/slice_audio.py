
from __future__ import annotations

"""
slice_audio.py

Goal:
  Traverse a nested audio dataset under `Data/raw/`, slice audio into fixed 5-second
  chunks, convert each chunk into a Mel-spectrogram, and save the result as PNGs
  under `Data/spectrograms/` while preserving the folder structure.

Key behaviors:
  - Discovery uses `os.walk`.
  - Audio is loaded with librosa at 22050 Hz (mono).
  - Chunks are non-overlapping; the final remainder < 5 seconds is dropped.
  - Output naming includes chunk start time: `<stem>_start{N}s.png`.
  - Progress bar is grouped by "pond" (top-level folder under `raw/`).
"""

import os
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import matplotlib

# Use a non-interactive backend so this can run on servers/CI without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# Update this set if you add more formats later.
SUPPORTED_EXTS = {".wav", ".mp3"}


@dataclass(frozen=True)
class Config:
    """Pipeline settings (all together so it's easy to tweak)."""

    raw_root: Path
    out_root: Path
    sample_rate: int = 22050
    chunk_seconds: int = 5
    n_mels: int = 128
    fmin: int = 0
    fmax: int = 8000
    # Frequency range considered for the Mel-spectrogram.
    #
    # Note: For Litoria aurea (Green and Golden Bell Frog), you may get cleaner
    # spectrograms by focusing on a narrower band, e.g.:
    #   --fmin 400 --fmax 4000
    # This reduces low-frequency wind/water thumps and high-frequency insect noise.


def iter_audio_files_os_walk(raw_root: Path) -> Iterable[Path]:
    """
    Find all audio files under raw_root using os.walk.

    Returns pathlib Paths for convenience everywhere else.
    """
    for dirpath, _, filenames in os.walk(raw_root):
        d = Path(dirpath)
        for name in filenames:
            p = d / name
            if p.suffix.lower() in SUPPORTED_EXTS:
                yield p


def pond_name_for_path(raw_root: Path, audio_path: Path) -> str:
    """
    "Pond" = top-level folder under raw_root.

    Example:
      raw_root=Data/raw
      audio_path=Data/raw/1A/December Check/Data/foo.wav  -> pond "1A"
    """
    rel = audio_path.relative_to(raw_root)
    return rel.parts[0] if len(rel.parts) > 0 else "raw"


def save_mel_png(
    y_chunk: np.ndarray,
    *,
    sr: int,
    out_path: Path,
    n_mels: int,
    fmin: int,
    fmax: int,
) -> None:
    """
    Convert a waveform chunk into a Mel-spectrogram PNG.

    Notes:
      - We save an axis-free image to keep the output clean for ML ingestion.
      - If you want consistent pixel sizes across all files, keep figsize/dpi fixed
        and avoid bbox_inches="tight".
    """
    S = librosa.feature.melspectrogram(
        y=y_chunk,
        sr=sr,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Minimal, axis-free image for ML pipelines.
    fig = plt.figure(figsize=(3.2, 3.2), dpi=150)
    ax = fig.add_subplot(111)
    ax.imshow(S_db, origin="lower", aspect="auto", interpolation="nearest")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def process_file(cfg: Config, audio_path: Path) -> int:
    """
    Process one audio file:
      load -> slice -> mel -> png

    Returns the number of PNG chunks written.
    """
    y, sr = librosa.load(audio_path, sr=cfg.sample_rate, mono=True)
    samples_per_chunk = int(sr * cfg.chunk_seconds)
    if samples_per_chunk <= 0:
        return 0

    # Preserve the subfolder structure from raw/ into spectrograms/.
    # Example:
    #   Data/raw/1A/x/y.wav -> Data/spectrograms/1A/x/y_start0s.png
    rel = audio_path.relative_to(cfg.raw_root)
    out_dir = cfg.out_root / rel.parent

    written = 0
    # Non-overlapping fixed 5-second chunks; drop remainder shorter than 5s.
    # If you prefer padding the last chunk instead, change the loop bounds and pad y_chunk.
    for i, start in enumerate(range(0, len(y) - samples_per_chunk + 1, samples_per_chunk)):
        y_chunk = y[start : start + samples_per_chunk]
        # Start time is deterministic because chunking is non-overlapping.
        start_s = i * cfg.chunk_seconds
        out_name = f"{audio_path.stem}_start{start_s}s.png"
        out_path = out_dir / out_name
        save_mel_png(
            y_chunk,
            sr=sr,
            out_path=out_path,
            n_mels=cfg.n_mels,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
        )
        written += 1

    return written


def _resolve_path(repo_root: Path, p: Path) -> Path:
    """Resolve relative paths against the repository root for convenience."""
    return p if p.is_absolute() else (repo_root / p)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    # Defaults match this repo’s layout:
    # - Data/raw/
    # - Data/spectrograms/
    default_raw = repo_root / "Data" / "raw"
    default_out = repo_root / "Data" / "spectrograms"

    p = argparse.ArgumentParser(
        description="Slice raw audio into fixed chunks and save Mel-spectrogram PNGs while preserving folder structure."
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=default_raw,
        help='Input root containing nested audio files (default: "Data/raw"). Example: --raw-root "Data/good data"',
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=default_out,
        help='Output root for spectrogram PNGs (default: "Data/spectrograms").',
    )
    p.add_argument(
        "--out-subdir",
        type=str,
        default="",
        help='Optional subfolder under out-root (e.g. "good_data" -> Data/spectrograms/good_data).',
    )
    p.add_argument("--sample-rate", type=int, default=22050, help="Resample audio to this rate (Hz).")
    p.add_argument("--chunk-seconds", type=int, default=5, help="Chunk length in seconds.")
    p.add_argument("--n-mels", type=int, default=128, help="Number of Mel bins.")
    p.add_argument("--fmin", type=int, default=0, help="Minimum frequency (Hz) for Mel-spectrogram.")
    p.add_argument("--fmax", type=int, default=8000, help="Maximum frequency (Hz) for Mel-spectrogram.")
    p.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="Process only the first N audio files per pond (0 = no limit). Useful for quick demos.",
    )
    args = p.parse_args()

    cfg = Config(
        raw_root=_resolve_path(repo_root, args.raw_root),
        out_root=_resolve_path(repo_root, args.out_root) / (args.out_subdir or ""),
        sample_rate=int(args.sample_rate),
        chunk_seconds=int(args.chunk_seconds),
        n_mels=int(args.n_mels),
        fmin=int(args.fmin),
        fmax=int(args.fmax),
    )

    if not cfg.raw_root.exists():
        raise FileNotFoundError(f"raw_root not found: {cfg.raw_root}")

    # Group by pond (top-level folder under raw) so tqdm can show which pond is running.
    pond_to_files: dict[str, list[Path]] = defaultdict(list)
    for p in iter_audio_files_os_walk(cfg.raw_root):
        pond_to_files[pond_name_for_path(cfg.raw_root, p)].append(p)

    if not pond_to_files:
        print(f"No audio files found under {cfg.raw_root} (supported: {sorted(SUPPORTED_EXTS)})")
        return 0

    total_written = 0
    ponds = sorted(pond_to_files.keys())
    for pond in ponds:
        files = sorted(pond_to_files[pond])
        if args.limit_files and args.limit_files > 0:
            files = files[: int(args.limit_files)]
        bar = tqdm(files, desc=f"Processing pond {pond}", unit="file")
        for audio_path in bar:
            try:
                total_written += process_file(cfg, audio_path)
            except Exception as e:
                # Keep the run going; surface failures clearly.
                bar.write(f"[ERROR] {audio_path}: {e}")

    print(f"Done. Wrote {total_written} spectrogram PNG(s) to {cfg.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
 
 