"""One analysis: read a detection record, aggregate it, and write every output."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from frog_classifier.files import replace_file
from frog_classifier.inference.store import DATABASE_FILE, DetectionStore
from frog_classifier.training.artifact import write_csv

from .activity import Activity, ActivityBin, read_activity
from .figures import (
    BY_DATE_FIGURE,
    CLOCK_FIGURE,
    TIME_OF_DAY_FIGURE,
    write_by_date_figure,
    write_clock_figure,
    write_time_of_day_figure,
)
from .report import render_report


REPORT_FILE = "report.md"
HOUR_TABLE = "activity_by_hour.csv"
DATE_TABLE = "activity_by_date.csv"
WINDOWS_TABLE = "windows.csv"
OUTPUT_FILES = (
    REPORT_FILE, TIME_OF_DAY_FIGURE, CLOCK_FIGURE, BY_DATE_FIGURE, HOUR_TABLE, DATE_TABLE, WINDOWS_TABLE,
)
_BIN_COLUMNS = ("recorded_windows", "recorded_minutes", "recordings", "detected_windows", "calling_share")
_WINDOW_COLUMNS = ("recording_path", "window_start_s", "date", "hour", "probability", "detected")


def default_analysis_dir(results_root: Path, record_dir: Path) -> Path:
    return results_root / record_dir.resolve().name


def analyze_record(
    record_dir: Path,
    output_dir: Path,
    *,
    threshold: float | None = None,
    utc_offset: str | None = None,
) -> Activity:
    """Analyze one detection record into output_dir, replacing any earlier analysis there.

    The record is opened read-only and never changes.
    """
    with DetectionStore.open_read_only(record_dir / DATABASE_FILE) as store:
        activity = read_activity(store, threshold=threshold, utc_offset=utc_offset)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_table(output_dir / HOUR_TABLE, ("hour", *_BIN_COLUMNS), (
        {"hour": hour, **_bin_values(bin_)} for hour, bin_ in enumerate(activity.by_hour)
    ))
    _write_table(output_dir / DATE_TABLE, ("date", *_BIN_COLUMNS), (
        {"date": day.isoformat(), **_bin_values(bin_)} for day, bin_ in activity.by_date.items()
    ))
    _write_table(output_dir / WINDOWS_TABLE, _WINDOW_COLUMNS, (
        {
            "recording_path": window.recording_path,
            "window_start_s": window.window_start_s,
            "date": window.date.isoformat(),
            "hour": window.hour,
            "probability": f"{window.probability:.8f}",
            "detected": int(activity.is_detected(window)),
        }
        for window in activity.windows
    ))
    for name, write in (
        (TIME_OF_DAY_FIGURE, write_time_of_day_figure),
        (CLOCK_FIGURE, write_clock_figure),
        (BY_DATE_FIGURE, write_by_date_figure),
    ):
        replace_file(output_dir / name, lambda staging, write=write: write(staging, activity))
    report = render_report(activity, record_dir.resolve().name).encode("utf-8")
    replace_file(output_dir / REPORT_FILE, lambda staging: staging.write_bytes(report))
    return activity


def _bin_values(bin_: ActivityBin) -> dict[str, object]:
    return {
        "recorded_windows": bin_.recorded_windows,
        "recorded_minutes": f"{bin_.recorded_minutes:.2f}",
        "recordings": bin_.recordings,
        "detected_windows": bin_.detected_windows,
        "calling_share": None if bin_.calling_share is None else f"{bin_.calling_share:.6f}",
    }


def _write_table(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    replace_file(path, lambda staging: write_csv(staging, columns, rows))
