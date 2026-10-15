from __future__ import annotations

import sqlite3
import tempfile
import unittest
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from frog_classifier.inference.detect import (
    DetectionError,
    DetectionSummary,
    RecordingOutcome,
    default_output_dir,
    detect_calls,
)
from frog_classifier.inference.export import DETECTIONS_CSV, RECORDINGS_CSV
from frog_classifier.inference.store import DATABASE_FILE, StoreError
from frog_classifier.preprocessing import decode_png, render_recording
from frog_classifier.training.fit import predict_probabilities
from frog_classifier.training.saved import load_saved_run
from tests.inference.helpers import AUDIOMOTH_COMMENT, build_saved_run, write_recording
from tests.preprocessing.helpers import noise


CPU = torch.device("cpu")


class Clock:
    """Each reading is one second after the last, so every stored time is distinct."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


class DetectCallsTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.run_dir = build_saved_run(self.root / "models", threshold=0.5)
        self.input_root = self.root / "raw" / "site"
        write_recording(
            self.input_root / "20220123_010000.WAV",
            noise(12, sample_rate_hz=48000, seed=1),
            comment=AUDIOMOTH_COMMENT,
        )
        write_recording(self.input_root / "20220123_020000.WAV", noise(10, sample_rate_hz=48000, seed=2))
        # The same file name in another folder is a different recording.
        write_recording(
            self.input_root / "nested" / "20220123_010000.WAV",
            noise(5, sample_rate_hz=22050, seed=3),
            sample_rate_hz=22050,
        )
        write_recording(self.input_root / "20220124_010000.wav", noise(3, sample_rate_hz=48000, seed=4))
        self.output = self.root / "results" / "detections" / "site"

    def _detect(self, output: Path | None = None, *, workers: int = 1, **options) -> DetectionSummary:
        return detect_calls(
            self.run_dir,
            self.input_root,
            output or self.output,
            device=CPU,
            workers=workers,
            **options,
        )

    def _query(self, statement: str, parameters: tuple = (), *, output: Path | None = None) -> list[tuple]:
        connection = sqlite3.connect((output or self.output) / DATABASE_FILE)
        try:
            return connection.execute(statement, parameters).fetchall()
        finally:
            connection.close()

    def test_every_window_of_every_recording_is_stored_and_exported(self) -> None:
        summary = self._detect()

        self.assertEqual((summary.found, summary.already_done, summary.processed, summary.failed), (4, 0, 4, 0))
        self.assertEqual(self._query("SELECT path, windows, status, message FROM recordings ORDER BY path"), [
            ("20220123_010000.WAV", 2, "ok", None),
            ("20220123_020000.WAV", 2, "ok", None),
            ("20220124_010000.wav", 0, "ok", "shorter than one 5-second window"),
            ("nested/20220123_010000.WAV", 1, "ok", None),
        ])
        self.assertEqual(self._query(
            "SELECT started_at, utc_offset FROM recordings WHERE path = ?", ("20220123_010000.WAV",),
        ), [("2022-01-23T01:00:00", "+11:00")])
        self.assertEqual(summary.windows, 5)
        self.assertEqual(
            summary.detections,
            self._query("SELECT COUNT(*) FROM windows WHERE probability >= 0.5")[0][0],
        )
        self.assertAlmostEqual(summary.audio_seconds, 30.0, places=3)
        self.assertIsNone(summary.csv_problem)
        self.assertTrue((self.output / DETECTIONS_CSV).is_file())
        self.assertTrue((self.output / RECORDINGS_CSV).is_file())

    def test_probabilities_equal_predictions_on_the_slicer_images(self) -> None:
        self._detect()

        saved = load_saved_run(self.run_dir)
        for relative in ("20220123_010000.WAV", "20220123_020000.WAV", "nested/20220123_010000.WAV"):
            with self.subTest(recording=relative):
                chunks = render_recording(self.input_root / relative, saved.preprocessing)
                pixels = np.stack([decode_png(chunk.png_bytes) for chunk in chunks])
                stored = self._query("SELECT probability FROM windows WHERE path = ? ORDER BY start_s", (relative,))

                np.testing.assert_allclose(
                    [probability for (probability,) in stored],
                    predict_probabilities(saved.model, pixels, CPU),
                    rtol=0,
                    atol=1e-9,
                )

    def test_an_unreadable_recording_is_recorded_as_failed_and_retried_next_time(self) -> None:
        (self.input_root / "20220125_010000.wav").write_bytes(b"not audio")

        with warnings.catch_warnings():
            # librosa warns before trying its fallback decoder; the recorded failure is what is under test.
            warnings.simplefilter("ignore")
            first = self._detect()
            second = self._detect()

        self.assertEqual((first.processed, first.failed), (5, 1))
        self.assertEqual((second.already_done, second.processed, second.failed), (4, 1, 1))
        ((status, message),) = self._query(
            "SELECT status, message FROM recordings WHERE path = ?", ("20220125_010000.wav",),
        )
        self.assertEqual(status, "failed")
        self.assertTrue(message)

    def test_an_interrupted_run_resumes_without_repeating_finished_recordings(self) -> None:
        for workers in (1, 2):
            with self.subTest(workers=workers):
                output = self.root / f"interrupted-{workers}"
                stored: list[str] = []

                def interrupt_after_two(outcome: RecordingOutcome) -> None:
                    stored.append(outcome.path)
                    if len(stored) == 2:
                        raise KeyboardInterrupt

                with self.assertRaises(KeyboardInterrupt):
                    self._detect(
                        output,
                        workers=workers,
                        on_recording=interrupt_after_two,
                        clock=Clock(datetime(2026, 10, 15, tzinfo=timezone.utc)),
                    )
                before = self._query("SELECT path, processed_at FROM recordings ORDER BY path", output=output)

                resumed = self._detect(output, workers=workers, clock=Clock(datetime(2026, 10, 16, tzinfo=timezone.utc)))

                self.assertEqual([path for path, _ in before], sorted(stored))
                self.assertEqual((resumed.already_done, resumed.processed), (2, 2))
                after = dict(self._query("SELECT path, processed_at FROM recordings", output=output))
                for path, processed_at in before:
                    self.assertEqual(after[path], processed_at)
                self.assertEqual(self._query("SELECT COUNT(*) FROM windows", output=output), [(5,)])

    def test_parallel_rendering_stores_the_same_record_as_one_worker(self) -> None:
        one, two = self.root / "one-worker", self.root / "two-workers"
        self._detect(one, workers=1)
        self._detect(two, workers=2)

        recordings = (
            "SELECT path, recording_id, started_at, utc_offset, size_bytes, audio_sha256, "
            "duration_s, windows, status, message FROM recordings ORDER BY path"
        )
        windows = "SELECT path, start_s, probability FROM windows ORDER BY path, start_s"
        self.assertEqual(self._query(recordings, output=one), self._query(recordings, output=two))
        self.assertEqual(self._query(windows, output=one), self._query(windows, output=two))

    def test_a_folder_with_results_from_another_model_is_refused(self) -> None:
        self._detect()
        other = build_saved_run(self.root / "models", name="resnet18-20261016T000000Z", seed=1)

        with self.assertRaisesRegex(StoreError, "different model_run, model_sha256"):
            detect_calls(other, self.input_root, self.output, device=CPU, workers=1)

    def test_a_different_recording_under_a_stored_name_is_refused(self) -> None:
        self._detect()
        # Recorders on the same schedule write the same file names on every card.
        write_recording(self.input_root / "20220123_020000.WAV", noise(10, sample_rate_hz=48000, seed=9))

        with self.assertRaisesRegex(DetectionError, "differ from the recordings already stored.*20220123_020000.WAV"):
            self._detect()

    def test_an_output_folder_inside_the_input_folder_is_refused(self) -> None:
        with self.assertRaisesRegex(DetectionError, "must not lie inside the input folder"):
            self._detect(self.input_root / "results")

        self.assertFalse((self.input_root / "results").exists())

    def test_a_missing_input_folder_is_refused(self) -> None:
        with self.assertRaisesRegex(DetectionError, "input folder not found"):
            detect_calls(self.run_dir, self.root / "missing", self.output, device=CPU, workers=1)

    def test_the_default_output_folder_is_named_after_the_input_and_the_run(self) -> None:
        self.assertEqual(
            default_output_dir(self.root / "results", self.input_root, "resnet18-x"),
            self.root / "results" / "site-resnet18-x",
        )

    def test_a_drive_root_needs_an_output_folder_named_for_it(self) -> None:
        with self.assertRaisesRegex(DetectionError, "has no folder name"):
            default_output_dir(self.root / "results", Path(self.root.anchor), "resnet18-x")


if __name__ == "__main__":
    unittest.main()
