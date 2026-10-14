"""Render one recording into window images for detection.

Rendering runs in worker processes, so this module and everything it imports
stay free of PyTorch.
"""

from __future__ import annotations

import hashlib
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np

from frog_classifier.data.config import PreprocessingConfig
from frog_classifier.preprocessing import ChunkImage, load_waveform, render_waveform_images
from frog_classifier.training.images import expected_image_shape

from .recordings import RecordingTime, recording_time


_HASH_BLOCK_BYTES = 1 << 20


@dataclass(frozen=True)
class RenderJob:
    input_root: Path
    path: str
    config: PreprocessingConfig


@dataclass(frozen=True)
class RenderedRecording:
    path: str
    recording_id: str
    started_at: datetime | None
    utc_offset: str | None
    size_bytes: int | None
    audio_sha256: str | None
    duration_s: float | None
    window_starts: tuple[int, ...]
    images: np.ndarray
    notes: tuple[str, ...]
    error: str | None
    render_seconds: float


def start_render_worker() -> None:
    """Leave Ctrl+C to the main process, which cancels the remaining work and
    stops the workers itself."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def render_for_detection(job: RenderJob) -> RenderedRecording:
    """Read, hash, time, and render one recording.

    Any error while doing so becomes this recording's failure, so one bad file
    never stops a batch.
    """
    started = time.perf_counter()
    source = job.input_root / job.path
    size_bytes: int | None = None
    audio_sha256: str | None = None
    duration_s: float | None = None
    error: str | None = None
    timing = RecordingTime(None, None, ())
    chunks: list[ChunkImage] = []
    try:
        size_bytes = source.stat().st_size
        audio_sha256 = _file_sha256(source)
        timing = recording_time(source)
        waveform = load_waveform(source, job.config)
        duration_s = len(waveform) / job.config.audio.sample_rate_hz
        chunks = list(render_waveform_images(waveform, job.config))
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    notes = timing.notes
    if error is None and not chunks:
        notes += (f"shorter than one {job.config.audio.chunk_seconds}-second window",)
    if chunks:
        images = np.stack([chunk.pixels for chunk in chunks])
    else:
        images = np.empty((0, *expected_image_shape(job.config)), dtype=np.uint8)
    return RenderedRecording(
        path=job.path,
        recording_id=PurePosixPath(job.path).stem,
        started_at=timing.started_at,
        utc_offset=timing.utc_offset,
        size_bytes=size_bytes,
        audio_sha256=audio_sha256,
        duration_s=duration_s,
        window_starts=tuple(chunk.start_s for chunk in chunks),
        images=images,
        notes=notes,
        error=error,
        render_seconds=time.perf_counter() - started,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()
