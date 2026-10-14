from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from frog_classifier.training.baseline import baseline_features, fit_baseline


class BaselineTests(unittest.TestCase):
    def test_features_are_resized_scaled_and_flattened(self) -> None:
        images = [
            Image.fromarray(np.full((4, 6), 255, dtype=np.uint8)),
            Image.fromarray(np.zeros((4, 6), dtype=np.uint8)),
        ]

        features = baseline_features(images, size=2)

        self.assertEqual(features.shape, (2, 4))
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_array_equal(features[0], np.ones(4))
        np.testing.assert_array_equal(features[1], np.zeros(4))

    def test_fits_a_balanced_logistic_regression(self) -> None:
        features = np.array([[0.0], [0.1], [0.2], [0.9], [1.0]])
        labels = np.array([0, 0, 0, 1, 1])

        classifier = fit_baseline(features, labels, inverse_regularization=10.0)

        self.assertEqual(classifier.C, 10.0)
        self.assertEqual(classifier.class_weight, "balanced")
        np.testing.assert_array_equal(classifier.predict(features), labels)


if __name__ == "__main__":
    unittest.main()
