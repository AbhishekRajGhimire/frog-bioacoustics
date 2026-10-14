from __future__ import annotations

import unittest

from sklearn.metrics import average_precision_score, roc_auc_score

from frog_classifier.training.metrics import (
    HourCounts,
    RecallCount,
    counts_by_hour,
    evaluate,
    false_detections_per_hour,
    recall_by_faint,
    select_threshold,
)


class ThresholdTests(unittest.TestCase):
    def test_picks_the_threshold_with_the_highest_f1(self) -> None:
        # Thresholds 0.1, 0.35, 0.4, 0.8 give F1 0.67, 0.80, 0.50, 0.67.
        self.assertEqual(select_threshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.35)

    def test_ties_go_to_the_higher_threshold(self) -> None:
        # Thresholds 0.1 and 0.9 both give F1 2/3; nothing in between does better.
        self.assertEqual(select_threshold([1, 0, 0, 1], [0.1, 0.2, 0.3, 0.9]), 0.9)

    def test_needs_both_classes(self) -> None:
        with self.assertRaises(ValueError):
            select_threshold([1, 1], [0.2, 0.8])


class EvaluationTests(unittest.TestCase):
    def test_counts_and_rates_at_the_threshold(self) -> None:
        labels = [1, 1, 0, 0, 0]
        probabilities = [0.9, 0.2, 0.8, 0.1, 0.3]

        metrics = evaluate(labels, probabilities, 0.5)

        self.assertEqual((metrics.examples, metrics.frog, metrics.background), (5, 2, 3))
        self.assertEqual(
            (metrics.true_positives, metrics.false_positives, metrics.true_negatives, metrics.false_negatives),
            (1, 1, 2, 1),
        )
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)
        self.assertAlmostEqual(metrics.f1, 0.5)
        self.assertAlmostEqual(metrics.false_positive_rate, 1 / 3)
        self.assertAlmostEqual(metrics.average_precision, average_precision_score(labels, probabilities))
        self.assertAlmostEqual(metrics.roc_auc, roc_auc_score(labels, probabilities))

    def test_a_probability_equal_to_the_threshold_counts_as_frog(self) -> None:
        self.assertEqual(evaluate([1, 0], [0.5, 0.1], 0.5).true_positives, 1)

    def test_no_detections_give_zero_precision_instead_of_failing(self) -> None:
        metrics = evaluate([1, 0], [0.2, 0.1], 0.5)

        self.assertEqual((metrics.precision, metrics.recall, metrics.f1), (0.0, 0.0, 0.0))

    def test_needs_both_classes(self) -> None:
        with self.assertRaises(ValueError):
            evaluate([0, 0], [0.2, 0.8], 0.5)


class BreakdownTests(unittest.TestCase):
    def test_counts_errors_by_recording_hour_with_unknown_hours_last(self) -> None:
        rows = counts_by_hour(
            ["20220101_230000", "20220101_010000", "20220101_010000", "no_timestamp"],
            [0, 1, 0, 1],
            [0.1, 0.2, 0.9, 0.8],
            0.5,
        )

        self.assertEqual(rows, (
            HourCounts(hour=1, background=1, false_positives=1, frog=1, missed=1),
            HourCounts(hour=23, background=1, false_positives=0, frog=0, missed=0),
            HourCounts(hour=None, background=0, false_positives=0, frog=1, missed=0),
        ))

    def test_recall_is_split_by_the_faint_tag(self) -> None:
        result = recall_by_faint([1, 1, 1, 0], [True, False, False, False], [0.9, 0.2, 0.8, 0.9], 0.5)

        self.assertEqual(result, {"faint": RecallCount(1, 1), "other": RecallCount(2, 1)})

    def test_false_detections_per_hour_use_the_window_length(self) -> None:
        self.assertAlmostEqual(false_detections_per_hour(0.01, 5), 7.2)


if __name__ == "__main__":
    unittest.main()
