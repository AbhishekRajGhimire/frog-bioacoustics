from __future__ import annotations

import unittest
from pathlib import Path

from frog_classifier.labeling.queue import build_queue, hour_of_day, is_night


def clip(recording_id: str, start_s: int = 0) -> Path:
    return Path("queue") / f"{recording_id}_start{start_s}s.png"


class HourOfDayTests(unittest.TestCase):
    def test_reads_the_hour_from_a_timestamped_recording_id(self) -> None:
        self.assertEqual(hour_of_day("20220121_030000"), 3)
        self.assertEqual(hour_of_day("2MM03935_20251130_180000"), 18)

    def test_missing_timestamp_is_unknown_and_counts_as_daytime(self) -> None:
        self.assertIsNone(hour_of_day("pond_recording"))
        self.assertFalse(is_night("pond_recording"))

    def test_night_spans_seven_pm_to_seven_am(self) -> None:
        for hour in (19, 23, 0, 6):
            self.assertTrue(is_night(f"20220101_{hour:02d}0000"), hour)
        for hour in (7, 12, 18):
            self.assertFalse(is_night(f"20220101_{hour:02d}0000"), hour)


class BuildQueueTests(unittest.TestCase):
    def test_interleaves_nine_night_clips_with_one_daytime_clip(self) -> None:
        night = [clip(f"202201{day:02d}_220000") for day in range(1, 19)]
        day = [clip(f"202201{day:02d}_100000") for day in range(1, 5)]

        queue = build_queue(night + day, seed=1)

        pattern = [is_night(path.name) for path in queue]
        self.assertEqual(len(queue), 22)
        self.assertEqual(pattern[:10], [True] * 9 + [False])
        self.assertEqual(pattern[10:20], [True] * 9 + [False])
        self.assertEqual(pattern[20:], [False, False])

    def test_caps_clips_per_recording(self) -> None:
        clips = [clip("20220101_220000", start_s=5 * index) for index in range(12)]

        queue = build_queue(clips, seed=1, per_recording_cap=5)

        self.assertEqual(len(queue), 5)

    def test_same_seed_same_order_and_different_seed_different_order(self) -> None:
        clips = [clip(f"202201{day:02d}_220000") for day in range(1, 25)]

        first = build_queue(clips, seed=7)
        second = build_queue(clips, seed=7)
        third = build_queue(clips, seed=8)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_skips_files_that_do_not_follow_the_naming_contract(self) -> None:
        clips = [clip("20220101_220000"), Path("queue") / "notes.png"]

        self.assertEqual(build_queue(clips, seed=1), [clip("20220101_220000")])

    def test_rejects_invalid_cap_and_share(self) -> None:
        with self.assertRaisesRegex(ValueError, "cap"):
            build_queue([], seed=1, per_recording_cap=0)
        with self.assertRaisesRegex(ValueError, "share"):
            build_queue([], seed=1, daytime_share=1.0)


if __name__ == "__main__":
    unittest.main()
