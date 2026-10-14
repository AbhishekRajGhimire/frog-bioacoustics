from __future__ import annotations

import unittest

import numpy as np
import torch

from frog_classifier.training.model import SpectrogramResNet, to_model_input


class ModelTests(unittest.TestCase):
    def test_returns_one_logit_per_image(self) -> None:
        model = SpectrogramResNet(pretrained=False).eval()

        with torch.inference_mode():
            logits = model(torch.rand((3, 1, 32, 48)))

        self.assertEqual(logits.shape, (3,))

    def test_rejects_images_without_one_channel(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch, 1"):
            SpectrogramResNet(pretrained=False)(torch.rand((2, 3, 32, 48)))

    def test_a_reloaded_state_dictionary_reproduces_outputs(self) -> None:
        torch.manual_seed(0)
        original = SpectrogramResNet(pretrained=False).eval()
        copy = SpectrogramResNet(pretrained=False)
        copy.load_state_dict(original.state_dict())
        copy.eval()
        images = torch.rand((2, 1, 32, 48))

        with torch.inference_mode():
            self.assertTrue(torch.equal(original(images), copy(images)))

    def test_model_input_scales_eight_bit_pixels(self) -> None:
        pixels = np.array([[[0, 255], [51, 102]]], dtype=np.uint8)

        tensor = to_model_input(pixels)

        self.assertEqual(tensor.shape, (1, 1, 2, 2))
        self.assertEqual(tensor.dtype, torch.float32)
        torch.testing.assert_close(tensor[0, 0], torch.tensor([[0.0, 1.0], [0.2, 0.4]]))

    def test_model_input_rejects_other_types(self) -> None:
        with self.assertRaises(ValueError):
            to_model_input(np.zeros((1, 2, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
