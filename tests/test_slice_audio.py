from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from frog_classifier.preprocessing.spectrogram import decode_png
from tests.data.helpers import load_test_config
from tests.preprocessing.helpers import silence, tone, write_wav


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "slice_audio.py"


class SliceAudioTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw_root = self.root / "raw"
        self.out_root = self.root / "processed"
        self.module = self._load_module()

    def test_parser_exposes_no_preprocessing_overrides(self) -> None:
        parser = self.module.build_parser(REPO_ROOT)
        arguments = parser.parse_args([])

        for name in ("sample_rate", "chunk_seconds", "n_mels", "fmin", "fmax"):
            with self.subTest(option=name):
                self.assertFalse(hasattr(arguments, name))
        self.assertFalse(hasattr(self.module, "Config"))

    def test_writes_one_image_per_complete_chunk_preserving_folders(self) -> None:
        write_wav(
            self.raw_root / "pond" / "recording.wav",
            np.concatenate([silence(5), tone(1000.0, 7)]),
        )

        result, _ = self._run("--raw-root", str(self.raw_root), "--out-root", str(self.out_root))

        self.assertEqual(result, 0)
        written = sorted(path.name for path in (self.out_root / "pond").glob("*.png"))
        self.assertEqual(written, ["recording_start0s.png", "recording_start5s.png"])
        image = decode_png((self.out_root / "pond" / "recording_start5s.png").read_bytes())
        self.assertEqual(image.shape, (128, 216))

    def test_overwrites_an_existing_image_by_name(self) -> None:
        write_wav(self.raw_root / "pond" / "recording.wav", tone(1000.0, 5))
        stale = self.out_root / "pond" / "recording_start0s.png"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"stale")

        result, _ = self._run("--raw-root", str(self.raw_root), "--out-root", str(self.out_root))

        self.assertEqual(result, 0)
        self.assertNotEqual(stale.read_bytes(), b"stale")
        self.assertEqual(decode_png(stale.read_bytes()).shape, (128, 216))

    def test_continues_after_an_unreadable_file(self) -> None:
        broken = self.raw_root / "pond" / "broken.wav"
        broken.parent.mkdir(parents=True)
        broken.write_bytes(b"not audio")
        write_wav(self.raw_root / "pond" / "recording.wav", tone(1000.0, 5))

        result, output = self._run(
            "--raw-root", str(self.raw_root), "--out-root", str(self.out_root),
        )

        self.assertEqual(result, 0)
        self.assertIn("[ERROR]", output)
        self.assertIn("broken.wav", output)
        self.assertTrue((self.out_root / "pond" / "recording_start0s.png").is_file())

    def test_limit_files_bounds_each_pond(self) -> None:
        write_wav(self.raw_root / "pond" / "a.wav", tone(1000.0, 5))
        write_wav(self.raw_root / "pond" / "b.wav", tone(1000.0, 5))

        result, _ = self._run(
            "--raw-root", str(self.raw_root),
            "--out-root", str(self.out_root),
            "--limit-files", "1",
        )

        self.assertEqual(result, 0)
        written = sorted(path.name for path in (self.out_root / "pond").glob("*.png"))
        self.assertEqual(written, ["a_start0s.png"])

    def test_discovers_uppercase_wav_extension(self) -> None:
        write_wav(self.raw_root / "pond" / "recording.WAV", tone(1000.0, 5))

        result, _ = self._run("--raw-root", str(self.raw_root), "--out-root", str(self.out_root))

        self.assertEqual(result, 0)
        self.assertTrue((self.out_root / "pond" / "recording_start0s.png").is_file())

    def test_rejects_out_root_inside_labeled_directory(self) -> None:
        repo_root = self.root / "repo"
        load_test_config(repo_root)
        (repo_root / "labeled").mkdir(parents=True)
        write_wav(self.raw_root / "pond" / "recording.wav", tone(1000.0, 5))
        out_root = repo_root / "labeled" / "spectrograms"

        result, _ = self._run(
            "--raw-root", str(self.raw_root),
            "--out-root", str(out_root),
            repo_root=repo_root,
        )

        self.assertEqual(result, 2)
        self.assertFalse(list(out_root.rglob("*.png")))

    def _run(self, *argv: str, repo_root: Path | None = None) -> tuple[int, str]:
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            if repo_root is None:
                result = self.module.main(argv)
            else:
                result = self.module.main(argv, repo_root=repo_root)
        return result, output.getvalue()

    def _load_module(self):
        module_name = f"slice_audio_test_{id(self)}"
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
