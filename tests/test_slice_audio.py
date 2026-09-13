from __future__ import annotations

import dataclasses
import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "slice_audio.py"


class SliceAudioPreprocessingIntegrationTests(unittest.TestCase):
    def test_slicer_config_has_no_preprocessing_defaults(self) -> None:
        """The tracked TOML is the only source of preprocessing values.

        Stale dataclass defaults once diverged from the command-line defaults
        and hid which frequency band actually produced the spectrograms.
        """
        module = self._load_module()
        fields = dataclasses.fields(module.Config)
        defaulted = sorted(
            field.name
            for field in fields
            if field.default is not dataclasses.MISSING
            or field.default_factory is not dataclasses.MISSING
        )
        self.assertEqual(defaulted, [])

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
