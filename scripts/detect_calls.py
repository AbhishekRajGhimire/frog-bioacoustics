"""Detect frog calls in every recording beneath a folder, using a saved model run.

Results go to one output folder. detections.sqlite is the complete record, and
detections.csv and recordings.csv are spreadsheet copies rewritten at the end of
every run. Running the same command again continues an interrupted run and
retries recordings that failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm


INTERRUPTED = "Interrupted. Finished recordings are saved; run the same command again to continue."


def main() -> int:
    # Imported here rather than at the top: rendering workers re-run this file's
    # top-level imports when they start, and they must not load PyTorch.
    from frog_classifier.inference.detect import (
        DetectionError,
        default_output_dir,
        default_workers,
        detect_calls,
    )
    from frog_classifier.inference.store import StoreError
    from frog_classifier.training.fit import select_device
    from frog_classifier.training.saved import SavedRunError

    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("run", type=Path, help="Saved model run folder, for example models/resnet18-20261014T114432Z.")
    parser.add_argument("input", type=Path, help="Folder of recordings; .wav and .mp3 files are found in every subfolder.")
    parser.add_argument("--output", type=Path, default=None, help="Output folder. Default: results/detections/<input folder>-<run folder>.")
    parser.add_argument("--workers", type=int, default=default_workers(), help="Processes that render recordings. Default: %(default)s.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None, help="Default: the GPU when available.")
    args = parser.parse_args()

    run_dir = _resolve(repo_root, args.run)
    input_root = _resolve(repo_root, args.input)
    if args.output is not None:
        output_dir = _resolve(repo_root, args.output)
    else:
        output_dir = default_output_dir(repo_root / "results" / "detections", input_root, run_dir.name)
    try:
        device = select_device(args.device)
    except ValueError as error:
        return _cannot_start(error)

    progress = _Progress()
    try:
        summary = detect_calls(
            run_dir,
            input_root,
            output_dir,
            device=device,
            workers=args.workers,
            on_start=progress.start,
            on_recording=progress.advance,
        )
    except (DetectionError, SavedRunError, StoreError) as error:
        progress.close()
        return _cannot_start(error)
    except KeyboardInterrupt:
        progress.close()
        print(INTERRUPTED, file=sys.stderr)
        return 130
    progress.close()

    print(
        f"Recordings: {summary.found} found, {summary.already_done} already done, "
        f"{summary.processed} processed now, {summary.failed} failed."
    )
    print(
        f"Detections: {summary.detections} of {summary.windows} windows at or above "
        f"the threshold {summary.threshold:.3f}."
    )
    if summary.processed:
        print(
            f"Speed: {summary.audio_seconds / 3600:.1f} hours of audio in {summary.wall_seconds:.0f} s, "
            f"{summary.audio_seconds / summary.wall_seconds:.0f} times real time; rendering took "
            f"{summary.render_seconds:.0f} s across workers and prediction {summary.predict_seconds:.0f} s."
        )
    print(f"Results: {summary.output_dir}")
    if summary.csv_problem is not None:
        print(summary.csv_problem, file=sys.stderr)
    if summary.failed:
        print(
            f"{summary.failed} recordings failed; see recordings.csv. "
            "Run the same command again to retry them.",
            file=sys.stderr,
        )
    return 1 if summary.failed or summary.csv_problem is not None else 0


class _Progress:
    """A progress bar that appears once the number of recordings to process is known."""

    def __init__(self) -> None:
        self._bar: tqdm | None = None

    def start(self, pending: int, done: int) -> None:
        print(f"{pending} recordings to process; {done} already done.")
        self._bar = tqdm(total=pending, unit="recording")

    def advance(self, outcome) -> None:
        if outcome.failed:
            self._bar.write(f"FAILED {outcome.path}: {outcome.message}")
        self._bar.update(1)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _cannot_start(error: Exception) -> int:
    print(f"Cannot start: {error}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
