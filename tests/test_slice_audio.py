from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from frog_classifier.data.config import load_preprocessing_config


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "slice_audio.py"


class SliceAudioPreprocessingIntegrationTests(unittest.TestCase):
    def test_tracked_defaults_reach_librosa_and_matplotlib_exactly(self) -> None:
        module = self._load_module()
        self.assertTrue(
            hasattr(module, "build_parser"),
            "slice_audio must expose the parser used by main",
        )
        preprocessing = load_preprocessing_config(
            REPO_ROOT / "config" / "preprocessing.toml"
        )
        parser = module.build_parser(REPO_ROOT, preprocessing)
        defaults = parser.parse_args([])
        self.assertEqual(defaults.sample_rate, 22050)
        self.assertEqual(defaults.chunk_seconds, 5)
        self.assertEqual(defaults.n_mels, 128)
        self.assertEqual(defaults.fmin, 0)
        self.assertEqual(defaults.fmax, 8000)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_root = root / "raw"
            out_root = root / "processed"
            audio_path = raw_root / "pond" / "recording.wav"
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"")
            waveform = np.zeros(22050 * 5, dtype=np.float32)
            mel_spectrogram = np.array([[1.0]], dtype=np.float32)
            decibels = np.array([[0.0]], dtype=np.float32)
            figure = MagicMock()
            axis = figure.add_subplot.return_value

            with (
                patch.object(
                    module.librosa,
                    "load",
                    return_value=(waveform, 22050),
                ) as load,
                patch.object(
                    module.librosa.feature,
                    "melspectrogram",
                    return_value=mel_spectrogram,
                ) as melspectrogram,
                patch.object(
                    module.librosa,
                    "power_to_db",
                    return_value=decibels,
                ) as power_to_db,
                patch.object(module.plt, "figure", return_value=figure) as make_figure,
                patch.object(module.plt, "close") as close,
            ):
                result = module.main((
                    "--raw-root",
                    str(raw_root),
                    "--out-root",
                    str(out_root),
                ))

            self.assertEqual(result, 0)
            load.assert_called_once_with(audio_path, sr=22050, mono=True)
            mel_call = melspectrogram.call_args
            self.assertIsNotNone(mel_call)
            mel_kwargs = dict(mel_call.kwargs)
            np.testing.assert_array_equal(mel_kwargs.pop("y"), waveform)
            self.assertEqual(mel_kwargs, {
                "sr": 22050,
                "n_mels": 128,
                "fmin": 0,
                "fmax": 8000,
                "power": 2.0,
                "n_fft": 2048,
                "hop_length": 512,
            })
            power_to_db.assert_called_once_with(mel_spectrogram, ref=np.max)
            make_figure.assert_called_once_with(figsize=(3.2, 3.2), dpi=150)
            figure.add_subplot.assert_called_once_with(111)
            imshow_call = axis.imshow.call_args
            self.assertIsNotNone(imshow_call)
            np.testing.assert_array_equal(imshow_call.args[0], decibels)
            self.assertEqual(imshow_call.kwargs, {
                "origin": "lower",
                "aspect": "auto",
                "interpolation": "nearest",
            })
            axis.axis.assert_called_once_with("off")
            figure.savefig.assert_called_once_with(
                out_root / "pond" / "recording_start0s.png",
                bbox_inches="tight",
                pad_inches=0,
            )
            close.assert_called_once_with(figure)

    def _load_module(self):
        module_name = f"slice_audio_test_{id(self)}"
        specification = importlib.util.spec_from_file_location(
            module_name,
            SCRIPT_PATH,
        )
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)
        specification.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
