from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from frog_classifier.inference.store import RecordingRow, ScanIdentity
from frog_classifier.training.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    MODEL_FILE,
    PREPROCESSING_FILE,
    finish_run,
    start_run,
)
from frog_classifier.training.model import ARCHITECTURE, SpectrogramResNet
from tests.data.helpers import VALID_CONFIG


AUDIOMOTH_COMMENT = (
    "Recorded at 01:00:00 23/01/2022 (UTC+11) by AudioMoth 248D9B0260370525 "
    "at medium gain while battery was 4.0V and temperature was 20.4C."
)


def riff_chunk(chunk_id: bytes, data: bytes) -> bytes:
    """One RIFF chunk, padded to an even length as the format requires."""
    return chunk_id + struct.pack("<I", len(data)) + data + b"\0" * (len(data) % 2)


def write_recording(
    path: Path,
    waveform: np.ndarray,
    *,
    sample_rate_hz: int = 48000,
    comment: str | None = None,
    extra_chunks: bytes = b"",
) -> Path:
    """A 16-bit mono WAV laid out like an AudioMoth file: the format chunk, any
    extra chunks, an optional LIST INFO comment, then the samples."""
    samples = (np.clip(waveform, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate_hz, sample_rate_hz * 2, 2, 16)
    info = b""
    if comment is not None:
        info = riff_chunk(b"LIST", b"INFO" + riff_chunk(b"ICMT", comment.encode("ascii") + b"\0"))
    body = b"WAVE" + riff_chunk(b"fmt ", fmt) + extra_chunks + info + riff_chunk(b"data", samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    return path


SCAN_IDENTITY = ScanIdentity(
    input_root="D:/recordings/site-a",
    model_run="resnet18-20261014T114432Z",
    model_sha256="a" * 64,
    threshold=0.5,
    preprocessing_sha256="b" * 64,
    chunk_seconds=5,
)


def recording_row(path: str, **changes: object) -> RecordingRow:
    """A successfully processed two-window recording, with any fields replaced."""
    row = RecordingRow(
        path=path,
        recording_id=PurePosixPath(path).stem,
        started_at="2022-01-23T01:00:00",
        utc_offset="+11:00",
        size_bytes=1000,
        audio_sha256="c" * 64,
        duration_s=10.0,
        windows=2,
        status="ok",
        message=None,
        render_seconds=0.5,
        predict_seconds=0.01,
        processed_at="2026-10-15T00:00:00Z",
    )
    return replace(row, **changes)


def build_saved_run(
    models_root: Path,
    *,
    name: str = "resnet18-20261015T000000Z",
    threshold: float = 0.5,
    seed: int = 0,
) -> Path:
    """A saved run holding an untrained network: detection needs a valid run, not a good model."""
    torch.manual_seed(seed)
    staging = start_run(models_root, name)
    torch.save(SpectrogramResNet(pretrained=False).state_dict(), staging / MODEL_FILE)
    (staging / PREPROCESSING_FILE).write_bytes(VALID_CONFIG)
    return finish_run(staging, {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_name": name,
        "model": {"architecture": ARCHITECTURE, "input_shape": [1, 128, 216]},
        "threshold": {"value": threshold, "rule": "chosen by the test"},
    })
