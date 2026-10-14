from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


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
