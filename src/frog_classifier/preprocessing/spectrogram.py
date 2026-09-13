from __future__ import annotations

import io

import librosa
import numpy as np
from PIL import Image


def mel_decibels(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int,
    n_mels: int,
    fmin_hz: int,
    fmax_hz: int,
    power: float,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """Mel power spectrogram in absolute decibels with shape (n_mels, frames).

    The reference is 1.0 and no top_db clipping is applied, so values are
    comparable between a whole recording and any chunk of it.
    """
    mel_power = librosa.feature.melspectrogram(
        y=np.asarray(waveform, dtype=np.float32),
        sr=sample_rate_hz,
        n_mels=n_mels,
        fmin=fmin_hz,
        fmax=fmax_hz,
        power=power,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return librosa.power_to_db(mel_power, ref=1.0, top_db=None)


def noise_floor(mel_db: np.ndarray, percentile: int) -> np.ndarray:
    """One floor value per band: the requested percentile across all frames."""
    if mel_db.ndim != 2 or mel_db.shape[1] == 0:
        raise ValueError("mel_db must have shape (bands, frames) with at least one frame")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    return np.percentile(mel_db, percentile, axis=1)


def chunk_image(
    mel_db: np.ndarray,
    floor: np.ndarray,
    *,
    db_floor: float,
    db_ceiling: float,
) -> np.ndarray:
    """8-bit image of decibels above the floor with the lowest band at the bottom."""
    if db_ceiling <= db_floor:
        raise ValueError("db_ceiling must exceed db_floor")
    if floor.shape != (mel_db.shape[0],):
        raise ValueError("floor must hold exactly one value per band")
    relative = mel_db - floor[:, np.newaxis]
    unit = np.clip((relative - db_floor) / (db_ceiling - db_floor), 0.0, 1.0)
    pixels = np.rint(unit * 255.0).astype(np.uint8)
    return pixels[::-1, :]


def encode_png(image: np.ndarray) -> bytes:
    """Lossless 8-bit grayscale PNG bytes with no metadata chunks."""
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("image must be a two-dimensional uint8 array")
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(buffer, format="PNG")
    return buffer.getvalue()


def decode_png(png_bytes: bytes) -> np.ndarray:
    """Grayscale uint8 array from PNG bytes."""
    with Image.open(io.BytesIO(png_bytes)) as image:
        return np.array(image.convert("L"), dtype=np.uint8)
