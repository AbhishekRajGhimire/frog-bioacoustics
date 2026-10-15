from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from frog_classifier.analytics.activity import ActivityBin, ActivityError, Coverage
from tests.analytics.helpers import read_test_activity, recording


class ReadActivityTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.record = Path(directory.name) / "record"

    def test_effort_and_detections_are_counted_by_hour_of_day(self) -> None:
        activity = read_test_activity(self.record, [
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.9, 0.1, 0.6, 0.2)),
            recording("a/20220124_010000.WAV", "2022-01-24T01:00:00", (0.1, 0.1)),
        ])

        self.assertEqual(
            activity.by_hour[1],
            ActivityBin(recorded_windows=6, recorded_minutes=0.5, recordings=2, detected_windows=2),
        )
        self.assertAlmostEqual(activity.by_hour[1].calling_share, 1 / 3)
        self.assertEqual(
            activity.by_hour[0],
            ActivityBin(recorded_windows=0, recorded_minutes=0.0, recordings=0, detected_windows=0),
        )
        self.assertIsNone(activity.by_hour[0].calling_share)
        self.assertEqual(activity.total.detected_windows, 2)

    def test_every_date_between_the_first_and_last_is_present(self) -> None:
        activity = read_test_activity(self.record, [
            recording("a/20220120_010000.WAV", "2022-01-20T01:00:00", (0.9,)),
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.9, 0.1)),
        ])

        self.assertEqual(list(activity.by_date), [date(2022, 1, day) for day in (20, 21, 22, 23)])
        self.assertIsNone(activity.by_date[date(2022, 1, 21)].calling_share)
        self.assertEqual(activity.by_date[date(2022, 1, 23)].recorded_windows, 2)

    def test_windows_are_placed_in_one_fixed_offset(self) -> None:
        # The last recorder states standard time; its 00:00 is 01:00 at UTC+11:00.
        recordings = [
            recording("a/20220301_010000.WAV", "2022-03-01T01:00:00", (0.9,)),
            recording("a/20220302_010000.WAV", "2022-03-02T01:00:00", (0.9,)),
            recording("a/20220404_000000.WAV", "2022-04-04T00:00:00", (0.9,), utc_offset="+10:00"),
        ]

        default = read_test_activity(self.record / "default", recordings)
        standard = read_test_activity(self.record / "standard", recordings, utc_offset="+10:00")

        self.assertEqual((default.time_basis.label, default.by_hour[1].recorded_windows), ("UTC+11:00", 3))
        self.assertEqual((standard.time_basis.label, standard.by_hour[0].recorded_windows), ("UTC+10:00", 3))
        self.assertEqual(default.coverage.stated_offsets, ("+10:00", "+11:00"))

    def test_a_recording_that_crosses_midnight_counts_on_both_dates(self) -> None:
        activity = read_test_activity(self.record, [
            recording("a/20220123_235955.WAV", "2022-01-23T23:59:55", (0.9, 0.9)),
        ])

        self.assertEqual(activity.by_date[date(2022, 1, 23)].recorded_windows, 1)
        self.assertEqual(activity.by_date[date(2022, 1, 24)].recorded_windows, 1)
        self.assertEqual((activity.by_hour[23].recorded_windows, activity.by_hour[0].recorded_windows), (1, 1))
        self.assertEqual(
            [f"{window.start:%Y-%m-%d %H:%M:%S}" for window in activity.windows],
            ["2022-01-23 23:59:55", "2022-01-24 00:00:00"],
        )

    def test_recordings_whose_clock_was_never_set_are_left_out_and_counted(self) -> None:
        # The record was created at 2026-10-15T00:00:00Z.
        activity = read_test_activity(self.record, [
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.9, 0.1)),
            recording("b/19700101_000000.WAV", "1970-01-01T00:00:00", (0.9, 0.9)),
            recording("c/20261016_000000.WAV", "2026-10-16T00:00:00", (0.9,)),
            # Within 14 hours of the scan, so a recorder far ahead of UTC can have made it.
            recording("d/20261015_130000.WAV", "2026-10-15T13:00:00", (0.1,)),
        ])

        self.assertEqual(activity.coverage.recordings_with_impossible_time, 2)
        self.assertEqual(activity.coverage.unplaced_windows, 3)
        self.assertEqual(
            {window.recording_path for window in activity.windows},
            {"a/20220123_010000.WAV", "d/20261015_130000.WAV"},
        )

    def test_recordings_that_cannot_be_placed_and_failed_recordings_are_counted(self) -> None:
        activity = read_test_activity(self.record, [
            recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.9, 0.1)),
            recording("a/unnamed.wav", None, (0.9, 0.9), utc_offset=None),
            recording("b/20220123_010000.WAV", "2022-01-23T01:00:00", (0.1, 0.1), utc_offset=None),
            recording("a/broken.wav", None, (), utc_offset=None, status="failed", message="LibsndfileError: bad"),
        ])

        self.assertEqual(activity.coverage, Coverage(
            recordings=3,
            failed_recordings=1,
            recordings_without_time=1,
            recordings_with_impossible_time=0,
            recordings_without_offset=1,
            stated_offsets=("+11:00",),
            unplaced_windows=2,
        ))
        self.assertEqual(len(activity.windows), 4)
        self.assertEqual(activity.by_hour[1].recorded_windows, 4)

    def test_a_threshold_override_changes_detections_but_not_effort(self) -> None:
        recordings = [recording("a/20220123_010000.WAV", "2022-01-23T01:00:00", (0.55, 0.95, 0.2))]

        model = read_test_activity(self.record / "model", recordings)
        strict = read_test_activity(self.record / "strict", recordings, threshold=0.9)

        self.assertEqual((model.threshold, model.total.detected_windows), (0.5, 2))
        self.assertEqual((strict.threshold, strict.total.detected_windows), (0.9, 1))
        self.assertEqual(strict.total.recorded_windows, 3)
        self.assertEqual(
            [(row.threshold, row.detected_windows) for row in strict.thresholds],
            [(0.5, 2), (0.7, 1), (0.9, 1), (0.95, 1), (0.99, 0)],
        )

    def test_a_threshold_outside_zero_to_one_is_refused(self) -> None:
        with self.assertRaisesRegex(ActivityError, "between 0 and 1"):
            read_test_activity(self.record, [], threshold=1.5)


if __name__ == "__main__":
    unittest.main()
