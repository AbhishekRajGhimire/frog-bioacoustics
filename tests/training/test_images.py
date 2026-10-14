from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from frog_classifier.training.images import expected_image_shape, load_spectrograms
from tests.data.helpers import load_test_config


class ImageLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _write(self, name: str, image: Image.Image) -> Path:
        path = self.root / name
        image.save(path)
        return path

    def test_contract_shape_is_mels_by_centred_frames(self) -> None:
        self.assertEqual(expected_image_shape(load_test_config(self.root)), (128, 216))

    def test_loads_images_in_order_with_their_checksums(self) -> None:
        first = self._write("a.png", Image.fromarray(np.full((4, 6), 10, dtype=np.uint8)))
        second = self._write("b.png", Image.fromarray(np.arange(24, dtype=np.uint8).reshape(4, 6)))

        loaded = load_spectrograms([second, first], (4, 6))

        self.assertEqual(loaded.pixels.dtype, np.uint8)
        np.testing.assert_array_equal(loaded.pixels[0], np.arange(24).reshape(4, 6))
        np.testing.assert_array_equal(loaded.pixels[1], np.full((4, 6), 10))
        self.assertEqual(loaded.sha256, (
            hashlib.sha256(second.read_bytes()).hexdigest(),
            hashlib.sha256(first.read_bytes()).hexdigest(),
        ))

    def test_rejects_a_wrong_size_naming_the_file(self) -> None:
        path = self._write("small.png", Image.fromarray(np.zeros((3, 6), dtype=np.uint8)))

        with self.assertRaisesRegex(ValueError, "small.png"):
            load_spectrograms([path], (4, 6))

    def test_rejects_a_colour_image(self) -> None:
        path = self._write("colour.png", Image.new("RGB", (6, 4)))

        with self.assertRaisesRegex(ValueError, "grayscale"):
            load_spectrograms([path], (4, 6))


if __name__ == "__main__":
    unittest.main()
