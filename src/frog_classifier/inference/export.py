"""Spreadsheet copies of the detection record, rewritten from the database."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from frog_classifier.files import replace_file
from frog_classifier.training.artifact import write_csv

from .store import DetectionStore


DETECTIONS_CSV = "detections.csv"
RECORDINGS_CSV = "recordings.csv"
DETECTION_COLUMNS = (
    "recording_path", "recording_id", "window_start_s", "window_end_s", "window_start_time",
    "utc_offset", "probability", "threshold", "model_run", "model_sha256", "preprocessing_sha256",
)
RECORDING_COLUMNS = (
    "recording_path", "recording_id", "started_at", "utc_offset", "status", "windows",
    "detections", "duration_s", "audio_sha256", "message",
)


def export_csv(store: DetectionStore, output_dir: Path) -> None:
    """Rewrite detections.csv and recordings.csv from the database.

    A detection is a window whose probability is at or above the run's threshold.
    """
    identity = store.identity
    recordings = {row.path: row for row in store.recording_rows()}
    detected = store.windows(minimum_probability=identity.threshold)
    counts = Counter(path for path, _, _ in detected)
    _replace(output_dir / DETECTIONS_CSV, DETECTION_COLUMNS, (
        {
            "recording_path": path,
            "recording_id": recordings[path].recording_id,
            "window_start_s": start_s,
            "window_end_s": start_s + identity.chunk_seconds,
            "window_start_time": _clock_time(recordings[path].started_at, start_s),
            "utc_offset": recordings[path].utc_offset,
            "probability": f"{probability:.8f}",
            "threshold": f"{identity.threshold:.8f}",
            "model_run": identity.model_run,
            "model_sha256": identity.model_sha256,
            "preprocessing_sha256": identity.preprocessing_sha256,
        }
        for path, start_s, probability in detected
    ))
    _replace(output_dir / RECORDINGS_CSV, RECORDING_COLUMNS, (
        {
            "recording_path": row.path,
            "recording_id": row.recording_id,
            "started_at": row.started_at,
            "utc_offset": row.utc_offset,
            "status": row.status,
            "windows": row.windows,
            "detections": counts[row.path],
            "duration_s": None if row.duration_s is None else f"{row.duration_s:.3f}",
            "audio_sha256": row.audio_sha256,
            "message": row.message,
        }
        for row in recordings.values()
    ))


def _clock_time(started_at: str | None, offset_s: int) -> str | None:
    if started_at is None:
        return None
    return (datetime.fromisoformat(started_at) + timedelta(seconds=offset_s)).isoformat()


def _replace(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    replace_file(path, lambda staging: write_csv(staging, columns, rows))
