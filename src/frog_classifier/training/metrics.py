"""Threshold rule and evaluation metrics for frog detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from frog_classifier.labeling.queue import hour_of_day


THRESHOLD_RULE = "maximum F1 on the validation fold, ties to the higher threshold"


@dataclass(frozen=True)
class SplitMetrics:
    examples: int
    frog: int
    background: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    average_precision: float
    roc_auc: float


@dataclass(frozen=True)
class HourCounts:
    hour: int | None
    background: int
    false_positives: int
    frog: int
    missed: int


@dataclass(frozen=True)
class RecallCount:
    clips: int
    detected: int


def select_threshold(labels: ArrayLike, probabilities: ArrayLike) -> float:
    """The probability that maximizes F1, predicting frog at or above it."""
    labels, probabilities = _checked(labels, probabilities)
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    precision, recall = precision[:-1], recall[:-1]
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)
    # Thresholds ascend, so the last maximum is the highest threshold among ties.
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def evaluate(labels: ArrayLike, probabilities: ArrayLike, threshold: float) -> SplitMetrics:
    labels, probabilities = _checked(labels, probabilities)
    detected = probabilities >= threshold
    frog = labels == 1
    true_positives = int(np.sum(detected & frog))
    false_positives = int(np.sum(detected & ~frog))
    true_negatives = int(np.sum(~detected & ~frog))
    false_negatives = int(np.sum(~detected & frog))
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    return SplitMetrics(
        examples=len(labels),
        frog=int(frog.sum()),
        background=int((~frog).sum()),
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=_ratio(2 * precision * recall, precision + recall),
        false_positive_rate=_ratio(false_positives, false_positives + true_negatives),
        average_precision=float(average_precision_score(labels, probabilities)),
        roc_auc=float(roc_auc_score(labels, probabilities)),
    )


def counts_by_hour(
    recording_ids: Sequence[str],
    labels: ArrayLike,
    probabilities: ArrayLike,
    threshold: float,
) -> tuple[HourCounts, ...]:
    """Errors per recording hour, in hour order, with recordings lacking a timestamp last."""
    detected = np.asarray(probabilities, dtype=float) >= threshold
    counts: dict[int | None, list[int]] = {}
    for recording_id, label, hit in zip(recording_ids, np.asarray(labels), detected, strict=True):
        row = counts.setdefault(hour_of_day(recording_id), [0, 0, 0, 0])
        if label == 1:
            row[2] += 1
            row[3] += int(not hit)
        else:
            row[0] += 1
            row[1] += int(hit)
    order = sorted(counts, key=lambda hour: (hour is None, hour or 0))
    return tuple(HourCounts(hour, *counts[hour]) for hour in order)


def recall_by_faint(
    labels: ArrayLike,
    faint: ArrayLike,
    probabilities: ArrayLike,
    threshold: float,
) -> dict[str, RecallCount]:
    """Detected frog clips among those tagged faint and among the others."""
    frog = np.asarray(labels) == 1
    tagged = np.asarray(faint, dtype=bool)
    detected = np.asarray(probabilities, dtype=float) >= threshold
    result = {}
    for name, mask in (("faint", frog & tagged), ("other", frog & ~tagged)):
        result[name] = RecallCount(clips=int(mask.sum()), detected=int((mask & detected).sum()))
    return result


def false_detections_per_hour(false_positive_rate: float, chunk_seconds: int) -> float:
    """Expected false detections in one hour of recording that holds no frog call."""
    return false_positive_rate * 3600 / chunk_seconds


def _checked(labels: ArrayLike, probabilities: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or labels.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional and the same length")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("evaluation needs both frog and background examples")
    return labels, probabilities


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
