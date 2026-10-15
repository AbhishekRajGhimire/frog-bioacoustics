"""The results download: the analysis and the detection spreadsheets in one zip file."""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

from frog_classifier.analytics.analyze import analyze_record
from frog_classifier.inference.export import DETECTIONS_CSV, RECORDINGS_CSV


def results_zip(record_dir: Path, *, threshold: float | None = None, utc_offset: str | None = None) -> bytes:
    """A zip holding one folder named after the record: the analysis at this
    threshold and offset, and the detection spreadsheets."""
    folder = record_dir.resolve().name
    buffer = io.BytesIO()
    with tempfile.TemporaryDirectory() as temporary:
        analysis_dir = Path(temporary) / "analysis"
        analyze_record(record_dir, analysis_dir, threshold=threshold, utc_offset=utc_offset)
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(analysis_dir.iterdir()):
                archive.write(path, f"{folder}/{path.name}")
            for name in (DETECTIONS_CSV, RECORDINGS_CSV):
                if (record_dir / name).is_file():
                    archive.write(record_dir / name, f"{folder}/{name}")
    return buffer.getvalue()
