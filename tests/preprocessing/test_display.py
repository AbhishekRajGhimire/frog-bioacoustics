from __future__ import annotations

import unittest

import numpy as np

from frog_classifier.preprocessing.display import colorize
from frog_classifier.preprocessing.spectrogram import encode_png


class ColorizeTests(unittest.TestCase):
    def test_maps_grayscale_through_viridis(self) -> None:
        image = np.array([[0, 255], [128, 64]], dtype=np.uint8)

        rgb = colorize(encode_png(image))

        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(rgb[0, 0].tolist(), [68, 1, 84])
        self.assertEqual(rgb[0, 1].tolist(), [253, 231, 37])


if __name__ == "__main__":
    unittest.main()
