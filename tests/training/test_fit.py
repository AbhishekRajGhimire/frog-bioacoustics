from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from frog_classifier.training.config import load_training_config
from frog_classifier.training.fit import _batches, fit_model, predict_probabilities, select_device
from tests.training.helpers import TRAINING_CONFIG


def _images(count: int, frog: bool, seed: int) -> np.ndarray:
    pixels = np.random.default_rng(seed).integers(0, 40, (count, 32, 48), dtype=np.uint8)
    if frog:
        pixels[:, 12:16, :] = 220
    return pixels


class FitTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "training.toml"
        path.write_bytes(TRAINING_CONFIG)
        self.config = load_training_config(path)

    def _fit(self, train_count: int, messages: list[str]):
        half = train_count // 2
        train = np.concatenate([_images(half, True, 1), _images(train_count - half, False, 2)])
        labels = np.array([1] * half + [0] * (train_count - half))
        val = np.concatenate([_images(2, True, 3), _images(2, False, 4)])
        return fit_model(
            train, labels, val, np.array([1, 1, 0, 0]), self.config,
            db_floor=0.0, db_ceiling=30.0, device=torch.device("cpu"), progress=messages.append,
        )

    def test_trains_keeps_the_best_epoch_and_reports_progress(self) -> None:
        messages: list[str] = []

        result = self._fit(8, messages)

        self.assertIn(len(result.history), (1, 2))
        self.assertIn(result.best_epoch, [record.epoch for record in result.history])
        self.assertFalse(result.model.training)
        self.assertTrue(any(message.startswith("epoch") for message in messages))

    def test_drops_a_trailing_partial_batch_only_when_full_batches_exist(self) -> None:
        generator = torch.Generator().manual_seed(0)

        self.assertEqual([len(batch) for batch in _batches(9, 4, generator)], [4, 4])
        self.assertEqual([len(batch) for batch in _batches(3, 4, generator)], [3])
        self.assertEqual(sorted(torch.cat(_batches(8, 4, generator)).tolist()), list(range(8)))

    def test_rejects_fewer_than_two_training_images(self) -> None:
        with self.assertRaisesRegex(ValueError, "two images"):
            self._fit(1, [])

    def test_probabilities_are_in_input_order_and_batch_independent(self) -> None:
        result = self._fit(8, [])
        pixels = np.concatenate([_images(3, True, 9), _images(3, False, 10)])

        together = predict_probabilities(result.model, pixels, torch.device("cpu"))
        alone = np.array([
            predict_probabilities(result.model, pixels[index : index + 1], torch.device("cpu"))[0]
            for index in range(len(pixels))
        ])

        self.assertEqual(together.shape, (6,))
        self.assertTrue(((together >= 0) & (together <= 1)).all())
        np.testing.assert_allclose(together, alone, atol=1e-6)


class DeviceTests(unittest.TestCase):
    def test_falls_back_to_the_cpu_without_cuda(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(select_device(), torch.device("cpu"))

    def test_an_explicit_cuda_request_without_a_gpu_fails_clearly(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                select_device("cuda")

    def test_an_explicit_cpu_request_is_honoured(self) -> None:
        self.assertEqual(select_device("cpu"), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
