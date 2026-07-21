from __future__ import annotations

from pathlib import Path

from frog_classifier.data.config import PreprocessingConfig, load_preprocessing_config


VALID_CONFIG = b'''schema_version = 1

[audio]
sample_rate_hz = 22050
mono = true
chunk_seconds = 5
overlap_seconds = 0
drop_incomplete_final_chunk = true

[spectrogram]
kind = "mel"
n_mels = 128
fmin_hz = 0
fmax_hz = 8000
power = 2.0
n_fft = 2048
hop_length = 512

[rendering]
format = "png"
figure_width_inches = 3.2
figure_height_inches = 3.2
dpi = 150
axis_visible = false
interpolation = "nearest"

[classes]
litoria_aurea = 1
non_target = 0
'''


def load_test_config(repo_root: Path) -> PreprocessingConfig:
    config_path = repo_root / "config" / "preprocessing.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(VALID_CONFIG)
    return load_preprocessing_config(config_path)


def write_png(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path
