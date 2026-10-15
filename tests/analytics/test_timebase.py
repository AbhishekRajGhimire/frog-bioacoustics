from __future__ import annotations

import unittest
from datetime import date

from frog_classifier.analytics.timebase import (
    OffsetError,
    choose_time_basis,
    format_offset,
    most_common_offset,
    parse_offset,
    window_time,
)


class OffsetTests(unittest.TestCase):
    def test_offsets_parse_and_format_back_unchanged(self) -> None:
        for text in ("+11:00", "-03:30", "+00:00", "+05:45"):
            with self.subTest(offset=text):
                self.assertEqual(format_offset(parse_offset(text)), text)

    def test_malformed_offsets_are_rejected(self) -> None:
        for text in ("11:00", "+11", "+1100", "UTC+11", "+15:00", "+10:60"):
            with self.subTest(offset=text):
                with self.assertRaises(OffsetError):
                    parse_offset(text)

    def test_the_offset_most_recordings_state_is_chosen(self) -> None:
        self.assertEqual(most_common_offset(["+11:00", "+10:00", "+11:00", None]), "+11:00")
        self.assertEqual(most_common_offset(["+11:00", "+10:00"]), "+10:00")
        self.assertIsNone(most_common_offset([None, None]))


class TimeBasisTests(unittest.TestCase):
    def test_a_requested_offset_wins(self) -> None:
        basis = choose_time_basis("+10:00", ["+11:00"])

        self.assertEqual(basis.label, "UTC+10:00")
        self.assertEqual(format_offset(basis.zone), "+10:00")

    def test_the_recorders_offset_is_the_default(self) -> None:
        self.assertEqual(choose_time_basis(None, ["+11:00", "+11:00", "+10:00"]).label, "UTC+11:00")

    def test_without_any_stated_offset_times_stay_on_the_recorder_clock(self) -> None:
        basis = choose_time_basis(None, [None])

        moment = window_time("2022-01-23T01:00:00", None, 10, basis.zone)

        self.assertEqual(basis.label, "recorder clock, UTC offset unknown")
        self.assertEqual((moment.hour, moment.minute, moment.second), (1, 0, 10))


class WindowTimeTests(unittest.TestCase):
    def test_a_window_is_converted_from_its_recording_offset(self) -> None:
        moment = window_time("2022-04-04T01:00:00", "+11:00", 5, parse_offset("+10:00"))

        self.assertEqual((moment.date(), moment.hour, moment.minute, moment.second), (date(2022, 4, 4), 0, 0, 5))

    def test_a_window_can_fall_on_the_next_date(self) -> None:
        moment = window_time("2022-01-23T23:59:55", "+11:00", 5, parse_offset("+11:00"))

        self.assertEqual((moment.date(), moment.hour), (date(2022, 1, 24), 0))


if __name__ == "__main__":
    unittest.main()
