from __future__ import annotations

import io

import matplotlib
import numpy as np
from PIL import Image


def colorize(png_bytes: bytes) -> np.ndarray:
    """RGB uint8 view of a grayscale PNG through the viridis colour map.

    Storage stays grayscale; this exists only so the labeler shows the
    familiar colours.
    """
    with Image.open(io.BytesIO(png_bytes)) as image:
        grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    rgba = matplotlib.colormaps["viridis"](grayscale)
    return np.rint(rgba[:, :, :3] * 255.0).astype(np.uint8)
