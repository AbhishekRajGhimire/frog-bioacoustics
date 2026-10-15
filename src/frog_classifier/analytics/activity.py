"""Recording effort and detected activity by hour of day and by date.

The unit is the window. Effort is how many windows were recorded, activity is how
many of them hold a detection, and the calling share is their ratio, which is
undefined, not zero, where nothing was recorded.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

from frog_classifier.inference.store import DetectionStore, ScanIdentity

from .timebase import TimeBasis, choose_time_basis, window_time


# The threshold table always shows these beside the analysis threshold.
TABLE_THRESHOLDS = (0.5, 0.7, 0.9, 0.95, 0.99)


class ActivityError(ValueError):
    pass


@dataclass(frozen=True)
class PlacedWindow:
    recording_path: str
    window_start_s: int
    date: date
    hour: int
    probability: float


@dataclass(frozen=True)
class ActivityBin:
    recorded_windows: int
    recorded_minutes: float
    recordings: int
    detected_windows: int

    @property
    def calling_share(self) -> float | None:
        """Detected windows over recorded windows; None where nothing was recorded."""
        if self.recorded_windows == 0:
            return None
        return self.detected_windows / self.recorded_windows


@dataclass(frozen=True)
class ThresholdRow:
    threshold: float
    detected_windows: int
    calling_share: float | None


@dataclass(frozen=True)
class Coverage:
    recordings: int
    failed_recordings: int
    recordings_without_time: int
    recordings_without_offset: int
    stated_offsets: tuple[str, ...]
    unplaced_windows: int


@dataclass(frozen=True)
class Activity:
    identity: ScanIdentity
    record_created_at: str
    threshold: float
    time_basis: TimeBasis
    windows: tuple[PlacedWindow, ...]
    total: ActivityBin
    by_hour: tuple[ActivityBin, ...]
    by_date: dict[date, ActivityBin]
    thresholds: tuple[ThresholdRow, ...]
    coverage: Coverage

    def is_detected(self, window: PlacedWindow) -> bool:
        return window.probability >= self.threshold


def read_activity(
    store: DetectionStore,
    *,
    threshold: float | None = None,
    utc_offset: str | None = None,
) -> Activity:
    """Place every window of a detection record in time and aggregate it.

    The threshold defaults to the record's model threshold, and the time basis to
    the UTC offset most recordings state.
    """
    identity = store.identity
    threshold = identity.threshold if threshold is None else threshold
    if not 0.0 <= threshold <= 1.0:
        raise ActivityError(f"the threshold must be between 0 and 1, not {threshold}")
    rows = store.recording_rows()
    succeeded = [row for row in rows if row.status == "ok"]
    timed = {row.path: row for row in succeeded if row.started_at is not None}
    basis = choose_time_basis(utc_offset, (row.utc_offset for row in timed.values()))

    windows: list[PlacedWindow] = []
    unplaced = 0
    for path, start_s, probability in store.windows():
        row = timed.get(path)
        if row is None:
            unplaced += 1
            continue
        moment = window_time(row.started_at, row.utc_offset, start_s, basis.zone)
        windows.append(PlacedWindow(path, start_s, moment.date(), moment.hour, probability))

    def summarize(selected: Sequence[PlacedWindow]) -> ActivityBin:
        return ActivityBin(
            recorded_windows=len(selected),
            recorded_minutes=len(selected) * identity.chunk_seconds / 60,
            recordings=len({window.recording_path for window in selected}),
            detected_windows=sum(window.probability >= threshold for window in selected),
        )

    by_hour_windows: dict[int, list[PlacedWindow]] = defaultdict(list)
    by_date_windows: dict[date, list[PlacedWindow]] = defaultdict(list)
    for window in windows:
        by_hour_windows[window.hour].append(window)
        by_date_windows[window.date].append(window)
    by_date: dict[date, ActivityBin] = {}
    if by_date_windows:
        day, last = min(by_date_windows), max(by_date_windows)
        while day <= last:
            by_date[day] = summarize(by_date_windows.get(day, []))
            day += timedelta(days=1)

    return Activity(
        identity=identity,
        record_created_at=store.created_at,
        threshold=threshold,
        time_basis=basis,
        windows=tuple(windows),
        total=summarize(windows),
        by_hour=tuple(summarize(by_hour_windows.get(hour, [])) for hour in range(24)),
        by_date=by_date,
        thresholds=tuple(_threshold_row(windows, value) for value in sorted({*TABLE_THRESHOLDS, threshold})),
        coverage=Coverage(
            recordings=len(succeeded),
            failed_recordings=len(rows) - len(succeeded),
            recordings_without_time=len(succeeded) - len(timed),
            recordings_without_offset=sum(row.utc_offset is None for row in timed.values()),
            stated_offsets=tuple(sorted({row.utc_offset for row in timed.values() if row.utc_offset is not None})),
            unplaced_windows=unplaced,
        ),
    )


def _threshold_row(windows: Sequence[PlacedWindow], threshold: float) -> ThresholdRow:
    detected = sum(window.probability >= threshold for window in windows)
    return ThresholdRow(threshold, detected, detected / len(windows) if windows else None)
