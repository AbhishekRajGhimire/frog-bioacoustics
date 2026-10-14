"""Logistic-regression baseline: a pipeline sanity check and the bar a deeper model must clear."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression


def baseline_features(images: Iterable[Image.Image], *, size: int) -> np.ndarray:
    """Each image as grayscale, resized to size by size, scaled to [0, 1], and flattened to one row."""
    rows = [
        np.asarray(image.convert("L").resize((size, size)), dtype=np.float32).reshape(-1) / 255.0
        for image in images
    ]
    return np.stack(rows, axis=0)


def fit_baseline(features: np.ndarray, labels: np.ndarray, *, inverse_regularization: float) -> LogisticRegression:
    classifier = LogisticRegression(C=inverse_regularization, max_iter=2000, class_weight="balanced")
    return classifier.fit(features, labels)
