from __future__ import annotations

import io
import tempfile
import time
import unittest
import warnings
from pathlib import Path

from PIL import Image

from frog_classifier.analytics.figures import (
    write_by_date_figure,
    write_clock_figure,
    write_time_of_day_figure,
)
from tests.analytics.helpers import read_test_activity, recording


WRITERS = (write_time_of_day_figure, write_clock_figure, write_by_date_figure)


class FigureTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _assert_png(self, path: Path) -> None:
        with Image.open(path) as image:
            self.assertEqual(image.format, "PNG")
            self.assertGreater(image.width, 300)

    def _write_all(self, activity, name: str) -> None:
        for writer in WRITERS:
            with self.subTest(figure=writer.__name__):
                path = self.root / name.format(writer=writer.__name__)
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    writer(path, activity)
                self._assert_png(path)

    def test_every_figure_is_a_png_image_even_under_a_temporary_name(self) -> None:
        activity = read_test_activity(self.root / "record", [
            recording("a/20220120_010000.WAV", "2022-01-20T01:00:00", (0.9, 0.1)),
            recording("a/20220123_230000.WAV", "2022-01-23T23:00:00", (0.9, 0.9, 0.2)),
        ])

        self._write_all(activity, ".{writer}.png.partial")

    def test_figures_draw_when_nothing_could_be_placed_in_time(self) -> None:
        activity = read_test_activity(self.root / "record", [
            recording("a/unnamed.wav", None, (0.9,), utc_offset=None),
        ])

        self._write_all(activity, "{writer}.png")

    def test_a_record_spanning_decades_draws_its_date_figure_quickly(self) -> None:
        # A recorder whose clock was never set names its files 1970-01-01.
        activity = read_test_activity(self.root / "record", [
            recording("a/19700101_000000.WAV", "1970-01-01T00:00:00", (0.9,)),
            recording("a/20220120_010000.WAV", "2022-01-20T01:00:00", (0.9,)),
        ])
        path = self.root / "by_date.png"

        start = time.perf_counter()
        write_by_date_figure(path, activity)

        self.assertLess(time.perf_counter() - start, 10)
        self._assert_png(path)


    def test_figures_can_be_drawn_into_memory(self) -> None:
        activity = read_test_activity(self.root / "record", [
            recording("a/20220120_010000.WAV", "2022-01-20T01:00:00", (0.9, 0.1)),
        ])

        for writer in WRITERS:
            with self.subTest(figure=writer.__name__):
                buffer = io.BytesIO()
                writer(buffer, activity)
                buffer.seek(0)
                with Image.open(buffer) as image:
                    self.assertEqual(image.format, "PNG")


if __name__ == "__main__":
    unittest.main()
