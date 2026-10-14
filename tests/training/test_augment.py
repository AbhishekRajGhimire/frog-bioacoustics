from __future__ import annotations

import unittest

import torch

from frog_classifier.training.augment import augment_batch
from frog_classifier.training.config import AugmentationConfig


def _generator(seed: int = 1) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _augment(images: torch.Tensor, config: AugmentationConfig, seed: int = 1) -> torch.Tensor:
    return augment_batch(images, config, db_floor=0.0, db_ceiling=30.0, generator=_generator(seed))


class AugmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.images = torch.rand((6, 1, 16, 24), generator=_generator(0))

    def test_preserves_shape_type_and_range_without_touching_the_input(self) -> None:
        original = self.images.clone()

        augmented = _augment(self.images, AugmentationConfig(5, 4, 6.0))

        self.assertEqual(augmented.shape, self.images.shape)
        self.assertEqual(augmented.dtype, torch.float32)
        self.assertGreaterEqual(float(augmented.min()), 0.0)
        self.assertLessEqual(float(augmented.max()), 1.0)
        self.assertTrue(torch.equal(self.images, original))

    def test_the_same_seed_gives_the_same_batch(self) -> None:
        config = AugmentationConfig(5, 4, 6.0)

        self.assertTrue(torch.equal(_augment(self.images, config, 3), _augment(self.images, config, 3)))
        self.assertFalse(torch.equal(_augment(self.images, config, 3), _augment(self.images, config, 4)))

    def test_time_shift_rolls_each_image_along_time(self) -> None:
        augmented = _augment(self.images, AugmentationConfig(0, 0, 0.0))

        for original, shifted in zip(self.images, augmented):
            with self.subTest():
                self.assertTrue(any(
                    torch.equal(shifted, torch.roll(original, offset, dims=2))
                    for offset in range(original.shape[2])
                ))

    def test_attenuation_never_brightens_and_keeps_the_noise_floor(self) -> None:
        # Constant along time so the shift cannot move anything.
        rows = torch.rand((6, 1, 16, 1), generator=_generator(5))
        rows[:, :, :4] = 0.0
        rows[:, :, 4] = 1.0
        images = rows.expand(-1, -1, -1, 24).contiguous()

        augmented = _augment(images, AugmentationConfig(0, 0, 6.0))

        self.assertTrue(bool((augmented <= images + 1e-6).all()))
        self.assertTrue(bool((augmented[images == 0.0] == 0.0).all()))
        self.assertTrue(bool((augmented < images - 1e-3).any()))
        # A pixel at the 30 dB ceiling drops by at most the 6 dB limit.
        self.assertTrue(bool((augmented[:, :, 4] >= 24.0 / 30.0 - 1e-5).all()))

    def test_masks_set_whole_time_columns_and_mel_rows_to_the_floor(self) -> None:
        images = torch.ones((8, 1, 16, 24))

        augmented = _augment(images, AugmentationConfig(5, 3, 0.0))

        for image in augmented[:, 0]:
            zero_columns = (image == 0).all(dim=0)
            zero_rows = (image == 0).all(dim=1)
            with self.subTest():
                self.assertTrue(torch.equal(image == 0, zero_columns[None, :] | zero_rows[:, None]))
                self.assertLessEqual(int(zero_columns.sum()), 5)
                self.assertLessEqual(int(zero_rows.sum()), 3)
                for indices in (zero_columns.nonzero().flatten(), zero_rows.nonzero().flatten()):
                    if len(indices):
                        self.assertEqual(int(indices[-1] - indices[0]) + 1, len(indices))


if __name__ == "__main__":
    unittest.main()
