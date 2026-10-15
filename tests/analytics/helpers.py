from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

from frog_classifier.analytics.activity import Activity, read_activity
from frog_classifier.inference.store import DATABASE_FILE, DetectionStore, RecordingRow
from tests.inference.helpers import SCAN_IDENTITY, recording_row


def recording(
    path: str,
    started_at: str | None,
    probabilities: Sequence[float],
    *,
    utc_offset: str | None = "+11:00",
    **changes: object,
) -> tuple[RecordingRow, tuple[float, ...]]:
    """A stored recording with one five-second window per probability."""
    row = recording_row(
        path,
        started_at=started_at,
        utc_offset=utc_offset,
        windows=len(probabilities),
        duration_s=5.0 * len(probabilities),
        **changes,
    )
    return row, tuple(probabilities)


def build_record(
    record_dir: Path,
    recordings: Sequence[tuple[RecordingRow, tuple[float, ...]]],
    *,
    threshold: float = 0.5,
    **identity_changes: object,
) -> Path:
    """A detection record folder whose model threshold is threshold."""
    record_dir.mkdir(parents=True, exist_ok=True)
    identity = replace(SCAN_IDENTITY, threshold=threshold, **identity_changes)
    with DetectionStore.open(record_dir / DATABASE_FILE, identity, now="2026-10-15T00:00:00Z") as store:
        for row, probabilities in recordings:
            store.save(row, tuple(range(0, 5 * len(probabilities), 5)), probabilities)
    return record_dir


def read_test_activity(
    record_dir: Path,
    recordings: Sequence[tuple[RecordingRow, tuple[float, ...]]],
    *,
    model_threshold: float = 0.5,
    **options: object,
) -> Activity:
    """Build a record with the given model threshold and read its activity with options."""
    build_record(record_dir, recordings, threshold=model_threshold)
    with DetectionStore.open_read_only(record_dir / DATABASE_FILE) as store:
        return read_activity(store, **options)
