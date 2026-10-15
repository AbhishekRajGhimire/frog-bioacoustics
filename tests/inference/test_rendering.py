from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from frog_classifier.inference.rendering import RenderJob, render_for_detection
from frog_classifier.preprocessing import load_waveform, render_waveform_images
from tests.data.helpers import load_test_config
from tests.inference.helpers import AUDIOMOTH_COMMENT, write_recording
from tests.preprocessing.helpers import noise


class RenderForDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = load_test_config(self.root)
        self.input_root = self.root / "raw"

    def _render(self, path: str):
        return render_for_detection(RenderJob(self.input_root, path, self.config))

    def test_renders_every_window_with_the_slicer_pixels_and_describes_the_source(self) -> None:
        source = write_recording(
            self.input_root / "site" / "20220123_010000.WAV",
            noise(12, sample_rate_hz=48000),
            comment=AUDIOMOTH_COMMENT,
        )

        rendered = self._render("site/20220123_010000.WAV")

        expected = np.stack([
            image.pixels
            for image in render_waveform_images(load_waveform(source, self.config), self.config)
        ])
        self.assertIsNone(rendered.error)
        self.assertEqual(rendered.path, "site/20220123_010000.WAV")
        self.assertEqual(rendered.recording_id, "20220123_010000")
        self.assertEqual(rendered.window_starts, (0, 5))
        np.testing.assert_array_equal(rendered.images, expected)
        self.assertEqual(rendered.started_at, datetime(2022, 1, 23, 1, 0, 0))
        self.assertEqual(rendered.utc_offset, "+11:00")
        self.assertEqual(rendered.size_bytes, source.stat().st_size)
        self.assertEqual(rendered.audio_sha256, hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertAlmostEqual(rendered.duration_s, 12.0, places=3)
        self.assertEqual(rendered.notes, ())
        self.assertGreater(rendered.render_seconds, 0.0)

    def test_a_recording_shorter_than_one_window_has_no_windows_and_says_so(self) -> None:
        write_recording(self.input_root / "20220123_010000.wav", noise(3, sample_rate_hz=48000))

        rendered = self._render("20220123_010000.wav")

        self.assertIsNone(rendered.error)
        self.assertEqual(rendered.window_starts, ())
        self.assertEqual(rendered.images.shape, (0, 128, 216))
        self.assertEqual(rendered.notes, ("shorter than one 5-second window",))

    def test_an_unreadable_file_is_a_failure_not_an_exception(self) -> None:
        path = self.input_root / "20220123_010000.wav"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not audio")

        with warnings.catch_warnings():
            # librosa warns before trying its fallback decoder; the failure is what is under test.
            warnings.simplefilter("ignore")
            rendered = self._render("20220123_010000.wav")

        self.assertIsNotNone(rendered.error)
        self.assertEqual(rendered.window_starts, ())
        self.assertEqual(rendered.images.shape, (0, 128, 216))
        self.assertEqual(rendered.size_bytes, 9)
        self.assertEqual(rendered.started_at, datetime(2022, 1, 23, 1, 0, 0))

    def test_a_file_that_vanished_is_a_failure_with_no_size(self) -> None:
        rendered = self._render("gone/20220123_010000.wav")

        self.assertIn("FileNotFoundError", rendered.error)
        self.assertIsNone(rendered.size_bytes)
        self.assertIsNone(rendered.started_at)

    def test_rendering_workers_do_not_load_pytorch(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import sys, frog_classifier.inference.rendering; print('torch' in sys.modules)"],
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertEqual(result.stdout.strip(), "False")


# A main process that starts one rendering worker, reports the worker's process
# id, and then waits to be killed.
MAIN_PROCESS = """
import multiprocessing, os, time
from concurrent.futures import ProcessPoolExecutor
from frog_classifier.inference.rendering import start_render_worker

if __name__ == "__main__":
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context, initializer=start_render_worker) as executor:
        print(executor.submit(os.getpid).result(), flush=True)
        time.sleep(600)
"""


def _process_is_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT: still running
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class RenderWorkerTests(unittest.TestCase):
    def test_a_worker_exits_when_its_main_process_is_killed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "main_process.py"
            script.write_bytes(MAIN_PROCESS.encode("utf-8"))
            main = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
            try:
                worker = int(main.stdout.readline())
                self.assertTrue(_process_is_alive(worker))

                main.kill()
                main.wait()
                deadline = time.monotonic() + 15
                while _process_is_alive(worker) and time.monotonic() < deadline:
                    time.sleep(0.2)

                orphaned = _process_is_alive(worker)
                if orphaned:
                    os.kill(worker, signal.SIGTERM)
                self.assertFalse(orphaned, "the worker outlived its main process")
            finally:
                main.kill()
                main.stdout.close()


if __name__ == "__main__":
    unittest.main()
