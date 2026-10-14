from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from frog_classifier.training.report import CurveInput, render_report, write_precision_recall_figure
from tests.training.helpers import example_metadata


EM_DASH = chr(0x2014)


class ReportTests(unittest.TestCase):
    def test_report_states_results_threshold_and_errors(self) -> None:
        report = render_report(example_metadata(cross_validation=True))

        for expected in (
            "# Training run resnet18-20261014T000000Z",
            "| ResNet-18 | 0.900 | 0.900 | 0.900 | 0.900 | 0.600 |",
            "Threshold rule: maximum F1 on the validation fold, ties to the higher threshold.",
            "about 28.8 false detections per hour",
            "| Background | 48 | 2 |",
            "| Frog | 2 | 18 |",
            "| 01:00 | 30 | 2 | 15 | 1 |",
            "| unknown | 20 | 0 | 5 | 1 |",
            "| Test | 2 | 2 | 18 | 16 |",
            "## Cross-validation",
            "| ResNet-18 | 0.900 (0.020) |",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, report)
        self.assertNotIn(EM_DASH, report)

    def test_cross_validation_section_is_absent_when_skipped(self) -> None:
        report = render_report(example_metadata(cross_validation=False))

        self.assertNotIn("## Cross-validation", report)
        self.assertIn("Cross-validation was skipped", report)

    def test_writes_a_png_figure(self) -> None:
        labels = np.array([0, 0, 1, 1, 0, 1])
        panel = CurveInput(
            "Test", labels,
            resnet=np.array([0.1, 0.3, 0.8, 0.9, 0.2, 0.7]),
            baseline=np.array([0.4, 0.5, 0.6, 0.4, 0.3, 0.7]),
            resnet_threshold=0.7, baseline_threshold=0.5,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curves.png"

            write_precision_recall_figure(path, [panel, panel])

            self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
