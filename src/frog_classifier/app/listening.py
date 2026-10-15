"""Hear and see one five-second window, as the model saw it."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from frog_classifier.data.config import PreprocessingConfig, load_preprocessing_config
from frog_classifier.preprocessing import load_waveform, render_waveform_images
from frog_classifier.preprocessing.display import colorize
from frog_classifier.preprocessing.spectrogram import encode_png
from frog_classifier.training.artifact import PREPROCESSING_FILE


class ListeningError(ValueError):
    pass


@dataclass(frozen=True)
class WindowView:
    wav: bytes
    image: np.ndarray


def model_preprocessing(models_root: Path, model_run: str) -> PreprocessingConfig:
    """The preprocessing contract saved with a model run."""
    path = models_root / model_run / PREPROCESSING_FILE
    if not path.is_file():
        raise ListeningError(f"the model {model_run} is no longer in {models_root}, so its windows cannot be shown")
    return load_preprocessing_config(path)


def load_window(recording: Path, start_s: int, config: PreprocessingConfig) -> WindowView:
    """The window's sound, raised to full loudness, and the spectrogram the model saw."""
    if not recording.is_file():
        raise ListeningError(f"{recording} is not there; connect the card or drive that holds it to listen")
    waveform = load_waveform(recording, config)
    image = next((chunk for chunk in render_waveform_images(waveform, config) if chunk.start_s == start_s), None)
    if image is None:
        raise ListeningError(f"{recording} has no five-second window starting at {start_s} seconds")
    rate = config.audio.sample_rate_hz
    samples = waveform[start_s * rate:(start_s + config.audio.chunk_seconds) * rate]
    return WindowView(_wav_bytes(samples, rate), colorize(encode_png(image.pixels)))


def _wav_bytes(samples: np.ndarray, sample_rate_hz: int) -> bytes:
    """16-bit mono WAV, scaled so the loudest sample is at full scale and faint calls can be heard."""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    scaled = samples / peak if peak > 0 else samples
    pcm = (np.clip(scaled, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as sound:
        sound.setnchannels(1)
        sound.setsampwidth(2)
        sound.setframerate(sample_rate_hz)
        sound.writeframes(pcm)
    return buffer.getvalue()
