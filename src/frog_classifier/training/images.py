"""Load stored spectrogram images exactly as they are on disk."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from frog_classifier.data.config import PreprocessingConfig


@dataclass(frozen=True)
class LoadedImages:
    pixels: np.ndarray
    sha256: tuple[str, ...]


def expected_image_shape(config: PreprocessingConfig) -> tuple[int, int]:
    """(height, width) of every stored spectrogram under the preprocessing contract.

    The width follows librosa's centred framing: one frame per hop, plus one.
    """
    samples_per_chunk = config.audio.sample_rate_hz * config.audio.chunk_seconds
    return config.spectrogram.n_mels, 1 + samples_per_chunk // config.spectrogram.hop_length


def load_spectrograms(paths: Sequence[Path], shape: tuple[int, int]) -> LoadedImages:
    """Decode 8-bit grayscale PNGs into one (count, height, width) array, in order,
    with the SHA-256 of each file's bytes so a run records exactly what it read."""
    pixels = np.empty((len(paths), *shape), dtype=np.uint8)
    checksums: list[str] = []
    for index, path in enumerate(paths):
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            if image.mode != "L":
                raise ValueError(f"{path}: expected an 8-bit grayscale image, found mode {image.mode}")
            array = np.asarray(image)
        if array.shape != shape:
            raise ValueError(
                f"{path}: expected {shape[0]} by {shape[1]} pixels (height by width), "
                f"found {array.shape[0]} by {array.shape[1]}"
            )
        pixels[index] = array
        checksums.append(hashlib.sha256(data).hexdigest())
    return LoadedImages(pixels=pixels, sha256=tuple(checksums))
