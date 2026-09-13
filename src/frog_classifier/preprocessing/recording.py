from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import librosa
import numpy as np

from frog_classifier.data.config import PreprocessingConfig

from .spectrogram import chunk_image, encode_png, mel_decibels, noise_floor


@dataclass(frozen=True)
class RenderedChunk:
    start_s: int
    png_bytes: bytes


def load_waveform(path: Path, config: PreprocessingConfig) -> np.ndarray:
    """Mono waveform at the configured sample rate."""
    waveform, _ = librosa.load(
        path,
        sr=config.audio.sample_rate_hz,
        mono=config.audio.mono,
    )
    return np.asarray(waveform, dtype=np.float32)


def render_waveform(
    waveform: np.ndarray,
    config: PreprocessingConfig,
) -> Iterator[RenderedChunk]:
    """Yield one rendered chunk per complete window, in order.

    The noise floor comes from every frame of the whole waveform, including
    any incomplete tail. Each chunk's spectrogram is computed from the chunk's
    own samples so that a batch or streaming inference path renders the same
    pixels.
    """
    samples_per_chunk = config.audio.sample_rate_hz * config.audio.chunk_seconds
    if len(waveform) < samples_per_chunk:
        return
    floor = noise_floor(
        _mel_decibels(waveform, config),
        config.normalization.noise_floor_percentile,
    )
    complete_chunks = len(waveform) // samples_per_chunk
    for index in range(complete_chunks):
        start = index * samples_per_chunk
        chunk = waveform[start : start + samples_per_chunk]
        image = chunk_image(
            _mel_decibels(chunk, config),
            floor,
            db_floor=config.normalization.db_floor,
            db_ceiling=config.normalization.db_ceiling,
        )
        yield RenderedChunk(
            start_s=index * config.audio.chunk_seconds,
            png_bytes=encode_png(image),
        )


def render_recording(
    path: Path,
    config: PreprocessingConfig,
) -> Iterator[RenderedChunk]:
    """Render every complete chunk of one recording file."""
    return render_waveform(load_waveform(path, config), config)


def _mel_decibels(waveform: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    spectrogram = config.spectrogram
    return mel_decibels(
        waveform,
        sample_rate_hz=config.audio.sample_rate_hz,
        n_mels=spectrogram.n_mels,
        fmin_hz=spectrogram.fmin_hz,
        fmax_hz=spectrogram.fmax_hz,
        power=spectrogram.power,
        n_fft=spectrogram.n_fft,
        hop_length=spectrogram.hop_length,
    )
