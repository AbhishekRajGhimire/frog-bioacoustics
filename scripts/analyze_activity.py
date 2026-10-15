"""Summarize calling activity in a detection record by hour of day and by date.

The analysis goes to one folder: a report, three figures, and tables that trace
every figure value back to individual windows. Recording effort is always shown
beside detected calls, so hours that were never recorded cannot look silent.
Running the command again replaces those files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from frog_classifier.analytics.activity import ActivityError
from frog_classifier.analytics.analyze import REPORT_FILE, analyze_record, default_analysis_dir
from frog_classifier.analytics.report import summary_lines
from frog_classifier.analytics.timebase import OffsetError
from frog_classifier.files import FileInUseError
from frog_classifier.inference.store import StoreError


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "record",
        type=Path,
        help="Detection record folder, for example results/detections/external-resnet18-20261014T114432Z.",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Detection threshold. Default: the model's own threshold.")
    parser.add_argument(
        "--utc-offset",
        default=None,
        help="One fixed UTC offset for every time, such as +10:00; write a negative one as --utc-offset=-03:00. "
        "Default: the offset the recorders state.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output folder. Default: results/analytics/<record folder>.")
    args = parser.parse_args()

    record_dir = _resolve(repo_root, args.record)
    if args.output is not None:
        output_dir = _resolve(repo_root, args.output)
    else:
        output_dir = default_analysis_dir(repo_root / "results" / "analytics", record_dir)
    try:
        activity = analyze_record(record_dir, output_dir, threshold=args.threshold, utc_offset=args.utc_offset)
    except (StoreError, OffsetError, ActivityError) as error:
        print(f"Cannot analyze: {error}", file=sys.stderr)
        return 2
    except FileInUseError as error:
        print(error, file=sys.stderr)
        return 1
    for line in summary_lines(activity):
        print(line)
    print(f"Report: {output_dir / REPORT_FILE}")
    return 0


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


if __name__ == "__main__":
    raise SystemExit(main())
