from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from frog_classifier.analytics.report import coverage_warnings, render_report, summary_lines
from tests.analytics.helpers import read_test_activity, recording
from tests.inference.helpers import SCAN_IDENTITY


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_the_report_states_coverage_unrecorded_time_thresholds_and_provenance(self) -> None:
        activity = read_test_activity(self.root / "record", [
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.9, 0.1)),
            recording("a/20220125_020000.WAV", "2022-01-25T02:00:00", (0.6, 0.6)),
        ])

        report = render_report(activity, "site-a")

        for expected in (
            "# Calling activity: site-a",
            "- Recorded 0.0 hours in 2 recordings on 2 of the 3 dates from 2022-01-23 to 2022-01-25.",
            "- Hours of day recorded: 01:00 to 02:59. "
            "Not recorded: 00:00 to 00:59, 03:00 to 23:59, so activity then is unknown.",
            "- Calls were detected in 75.0% of recorded time (3 of 4 windows of 5 seconds) at threshold 0.500.",
            "- Highest calling share: 02:00 to 02:59, 100.0% of recorded time.",
            "- Times are in UTC+11:00 for every recording, so daylight-saving changes cannot shift them.",
            "![Calling share by hour of day](time_of_day.png)",
            "| 00:00 | 0.00 | 0 | 0 | not recorded |",
            "| 01:00 | 0.00 | 1 | 1 | 50.0% |",
            "| 2022-01-24 | 0.00 | 0 | 0 | not recorded |",
            "| 0.990 | 0 | 0.0% |",
            f"| Model run | {SCAN_IDENTITY.model_run} |",
            "| Recordings without a start time, left out | 0 |",
        ):
            with self.subTest(line=expected):
                self.assertIn(expected, report)

    def test_times_without_any_stated_offset_are_the_recorder_clock(self) -> None:
        activity = read_test_activity(self.root / "record", [
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.9,), utc_offset=None),
        ])

        self.assertIn(
            "Times are the recorders' own clock, because no recording states its UTC offset.",
            summary_lines(activity),
        )

    def test_a_record_with_nothing_placed_in_time_says_so(self) -> None:
        activity = read_test_activity(self.root / "record", [
            recording("a/unnamed.wav", None, (0.9,), utc_offset=None),
        ])

        self.assertEqual(
            summary_lines(activity),
            [
                "No recording could be placed in time, so there is no activity to report.",
                "Left out of every result by time: 1 recording with no start time in the file name.",
            ],
        )
        self.assertIn("| Windows that could not be placed in time | 1 |", render_report(activity, "site-a"))

    def test_coverage_problems_are_warned_about(self) -> None:
        activity = read_test_activity(self.root / "record", [
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.9,)),
            recording("a/19700101_000000.WAV", "1970-01-01T00:00:00", (0.9,)),
            recording("a/19700101_000500.WAV", "1970-01-01T00:05:00", (0.9,)),
            recording("b/20220123_010000.WAV", "2022-01-23T01:00:00", (0.1,), utc_offset=None),
            recording("a/broken.wav", None, (), utc_offset=None, status="failed", message="LibsndfileError: bad"),
        ])

        self.assertEqual(coverage_warnings(activity), [
            "Left out of every result by time: 2 recordings with an impossible date, before 2005 or after the scan, "
            "probably because the recorder's clock was not set.",
            "Not included because reading failed: 1 recording. Scanning again retries failed recordings.",
            "Taken to use UTC+11:00 because no UTC offset is stated: 1 recording.",
        ])
        self.assertIn("| Recordings with an impossible start time, left out | 2 |", render_report(activity, "site-a"))


if __name__ == "__main__":
    unittest.main()
