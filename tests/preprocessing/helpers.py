from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile


SAMPLE_RATE_HZ = 22050
N_MELS = 128
FMIN_HZ = 400
FMAX_HZ = 4000


def tone(
    frequency_hz: float,
    seconds: float,
    *,
    amplitude: float = 0.5,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> np.ndarray:
    samples = int(round(seconds * sample_rate_hz))
    times = np.arange(samples, dtype=np.float64) / sample_rate_hz
    return (amplitude * np.sin(2.0 * np.pi * frequency_hz * times)).astype(np.float32)


def silence(seconds: float, *, sample_rate_hz: int = SAMPLE_RATE_HZ) -> np.ndarray:
    return np.zeros(int(round(seconds * sample_rate_hz)), dtype=np.float32)


def noise(
    seconds: float,
    *,
    amplitude: float = 0.1,
    seed: int = 0,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Seeded white noise so synthetic recordings have a realistic floor in every band."""
    samples = int(round(seconds * sample_rate_hz))
    return np.random.default_rng(seed).normal(0.0, amplitude, samples).astype(np.float32)


def write_wav(
    path: Path,
    waveform: np.ndarray,
    *,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, waveform, sample_rate_hz, subtype="PCM_16")
    return path


def mel_band_for(frequency_hz: float) -> int:
    """Index of the Mel band whose centre is closest to the frequency."""
    centres = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN_HZ, fmax=FMAX_HZ)
    return int(np.argmin(np.abs(centres - frequency_hz)))
