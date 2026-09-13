from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image

from frog_classifier.data import load_preprocessing_config
from frog_classifier.preprocessing import render_recording
from frog_classifier.preprocessing.spectrogram import decode_png, encode_png
from tests.preprocessing.helpers import silence, tone, write_wav


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_spectrograms.py"


class VerifySpectrogramsTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw_root = self.root / "raw"
        self.spectrogram_root = self.root / "processed" / "spectrograms"
        self.labeled_root = self.root / "labeled"
        self.module = self._load_module()
        config = load_preprocessing_config(REPO_ROOT / "config" / "preprocessing.toml")
        recording = write_wav(
            self.raw_root / "site" / "rec.wav",
            np.concatenate([silence(5), tone(1000.0, 5), silence(2)]),
        )
        chunks = {chunk.start_s: chunk.png_bytes for chunk in render_recording(recording, config)}
        self.queued = self._write(self.spectrogram_root / "site" / "rec_start0s.png", chunks[0])
        self.labeled = self._write(self.labeled_root / "litoria_aurea" / "rec_start5s.png", chunks[5])

    def test_consistent_tree_matches_everywhere(self) -> None:
        result, output = self._run("--all")

        self.assertEqual(result, 0)
        self.assertIn("MATCH", output)
        self.assertNotIn("DIFFERENT", output)
        self.assertIn("2 checked, 2 matched", output)

    def test_tampered_image_is_reported_and_fails(self) -> None:
        image = decode_png(self.labeled.read_bytes())
        image[0, 0] = 255 - image[0, 0]
        self.labeled.write_bytes(encode_png(image))

        result, output = self._run("--all")

        self.assertEqual(result, 2)
        self.assertIn(f"DIFFERENT {self.labeled}", output)

    def test_re_encoded_image_is_reported_as_encoding_drift(self) -> None:
        image = Image.open(self.labeled)
        image.load()
        image.save(self.labeled, format="PNG", compress_level=1)

        result, output = self._run("--all")

        self.assertEqual(result, 2)
        self.assertIn(f"ENCODING_DIFFERS {self.labeled}", output)

    def test_missing_recording_is_reported_and_fails(self) -> None:
        self._write(self.labeled_root / "non_target" / "ghost_start0s.png", b"x")

        result, output = self._run("--all")

        self.assertEqual(result, 2)
        self.assertIn("MISSING_RECORDING", output)
        self.assertIn("ghost_start0s.png", output)

    def test_ambiguous_recording_is_reported_and_fails(self) -> None:
        self._write(self.raw_root / "site" / "rec.mp3", b"")

        result, output = self._run("--all")

        self.assertEqual(result, 2)
        self.assertIn("AMBIGUOUS_RECORDING", output)

    def test_sample_below_one_is_rejected(self) -> None:
        result, output = self._run("--sample", "0")

        self.assertEqual(result, 2)
        self.assertIn("--sample", output)

    def test_chunk_beyond_recording_is_reported_and_fails(self) -> None:
        self._write(self.spectrogram_root / "site" / "rec_start30s.png", b"x")

        result, output = self._run("--all")

        self.assertEqual(result, 2)
        self.assertIn("MISSING_CHUNK", output)

    def test_sample_is_seeded_and_bounded(self) -> None:
        result, output = self._run("--sample", "1", "--seed", "3")

        self.assertEqual(result, 0)
        self.assertIn("2 checked, 2 matched", output)

    def _run(self, *extra: str) -> tuple[int, str]:
        argv = [
            "--raw-root", str(self.raw_root),
            "--spectrogram-root", str(self.spectrogram_root),
            "--labeled-root", str(self.labeled_root),
            *extra,
        ]
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = self.module.main(argv)
        return result, output.getvalue()

    def _write(self, path: Path, content: bytes) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _load_module(self):
        module_name = f"verify_spectrograms_test_{id(self)}"
        specification = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)
        specification.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
