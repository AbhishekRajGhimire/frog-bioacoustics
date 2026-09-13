from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


SUPPORTED_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class AudioConfig:
    sample_rate_hz: int
    mono: bool
    chunk_seconds: int
    overlap_seconds: int
    drop_incomplete_final_chunk: bool


@dataclass(frozen=True)
class SpectrogramConfig:
    kind: str
    n_mels: int
    fmin_hz: int
    fmax_hz: int
    power: float
    n_fft: int
    hop_length: int


@dataclass(frozen=True)
class NormalizationConfig:
    reference: str
    noise_floor_percentile: int
    db_floor: float
    db_ceiling: float


@dataclass(frozen=True)
class RenderingConfig:
    format: str
    bit_depth: int
    low_frequency_at_bottom: bool


@dataclass(frozen=True)
class PreprocessingConfig:
    schema_version: int
    audio: AudioConfig
    spectrogram: SpectrogramConfig
    normalization: NormalizationConfig
    rendering: RenderingConfig
    classes: Mapping[str, int]
    source_path: Path
    sha256: str


class ConfigError(ValueError):
    pass


def _section(data: dict[str, object], name: str, keys: set[str]) -> dict[str, object]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing or invalid {name} section")
    if set(value) != keys:
        raise ConfigError(f"invalid {name} keys: {sorted(value)}")
    return value


def _value(data: dict[str, object], key: str, expected: type) -> object:
    value = data[key]
    if expected is int and (not isinstance(value, int) or isinstance(value, bool)):
        raise ConfigError(f"{key} must be an integer")
    if expected is float and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise ConfigError(f"{key} must be numeric")
    if expected is float and not math.isfinite(float(value)):
        raise ConfigError(f"{key} must be finite")
    if expected is bool and not isinstance(value, bool):
        raise ConfigError(f"{key} must be a Boolean")
    if expected is str and not isinstance(value, str):
        raise ConfigError(f"{key} must be text")
    return value


def load_preprocessing_config(path: Path) -> PreprocessingConfig:
    source = path.read_bytes()
    try:
        data = tomllib.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"invalid preprocessing TOML: {error}") from error

    expected_top_level = {
        "schema_version", "audio", "spectrogram", "normalization", "rendering", "classes",
    }
    if set(data) != expected_top_level:
        missing = sorted(expected_top_level - set(data))
        unexpected = sorted(set(data) - expected_top_level)
        raise ConfigError(
            f"invalid top-level keys: {sorted(data)}; "
            f"missing: {missing}; unexpected: {unexpected}"
        )
    if _value(data, "schema_version", int) != SUPPORTED_SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version must be {SUPPORTED_SCHEMA_VERSION}; spectrograms rendered "
            "under an earlier schema must be regenerated"
        )

    audio_data = _section(data, "audio", {
        "sample_rate_hz", "mono", "chunk_seconds", "overlap_seconds",
        "drop_incomplete_final_chunk",
    })
    spectrogram_data = _section(data, "spectrogram", {
        "kind", "n_mels", "fmin_hz", "fmax_hz", "power", "n_fft", "hop_length",
    })
    normalization_data = _section(data, "normalization", {
        "reference", "noise_floor_percentile", "db_floor", "db_ceiling",
    })
    rendering_data = _section(data, "rendering", {
        "format", "bit_depth", "low_frequency_at_bottom",
    })
    classes_data = _section(data, "classes", {"litoria_aurea", "non_target"})

    audio = AudioConfig(
        sample_rate_hz=int(_value(audio_data, "sample_rate_hz", int)),
        mono=bool(_value(audio_data, "mono", bool)),
        chunk_seconds=int(_value(audio_data, "chunk_seconds", int)),
        overlap_seconds=int(_value(audio_data, "overlap_seconds", int)),
        drop_incomplete_final_chunk=bool(_value(audio_data, "drop_incomplete_final_chunk", bool)),
    )
    spectrogram = SpectrogramConfig(
        kind=str(_value(spectrogram_data, "kind", str)),
        n_mels=int(_value(spectrogram_data, "n_mels", int)),
        fmin_hz=int(_value(spectrogram_data, "fmin_hz", int)),
        fmax_hz=int(_value(spectrogram_data, "fmax_hz", int)),
        power=float(_value(spectrogram_data, "power", float)),
        n_fft=int(_value(spectrogram_data, "n_fft", int)),
        hop_length=int(_value(spectrogram_data, "hop_length", int)),
    )
    normalization = NormalizationConfig(
        reference=str(_value(normalization_data, "reference", str)),
        noise_floor_percentile=int(_value(normalization_data, "noise_floor_percentile", int)),
        db_floor=float(_value(normalization_data, "db_floor", float)),
        db_ceiling=float(_value(normalization_data, "db_ceiling", float)),
    )
    rendering = RenderingConfig(
        format=str(_value(rendering_data, "format", str)),
        bit_depth=int(_value(rendering_data, "bit_depth", int)),
        low_frequency_at_bottom=bool(_value(rendering_data, "low_frequency_at_bottom", bool)),
    )
    classes = {
        "litoria_aurea": int(_value(classes_data, "litoria_aurea", int)),
        "non_target": int(_value(classes_data, "non_target", int)),
    }

    if audio.sample_rate_hz <= 0 or audio.chunk_seconds <= 0:
        raise ConfigError("audio dimensions must be positive")
    if not 0 <= audio.overlap_seconds < audio.chunk_seconds:
        raise ConfigError("overlap must be nonnegative and shorter than the chunk")
    if not audio.mono or audio.overlap_seconds != 0 or not audio.drop_incomplete_final_chunk:
        raise ConfigError("audio settings must match the supported fixed-window contract")
    if not 0 <= spectrogram.fmin_hz < spectrogram.fmax_hz <= audio.sample_rate_hz / 2:
        raise ConfigError("frequency range must not exceed Nyquist")
    if min(
        spectrogram.n_mels,
        spectrogram.n_fft,
        spectrogram.hop_length,
        spectrogram.power,
    ) <= 0:
        raise ConfigError("spectrogram dimensions and power must be positive")
    if spectrogram.kind != "mel" or rendering.format != "png":
        raise ConfigError("only mel spectrograms rendered as png are supported")
    if normalization.reference != "recording":
        raise ConfigError("normalization reference must be recording")
    if not 0 <= normalization.noise_floor_percentile <= 100:
        raise ConfigError("noise_floor_percentile must be between 0 and 100")
    if normalization.db_ceiling <= normalization.db_floor:
        raise ConfigError("db_ceiling must exceed db_floor")
    if rendering.bit_depth != 8:
        raise ConfigError("bit_depth must be 8")
    if not rendering.low_frequency_at_bottom:
        raise ConfigError("low_frequency_at_bottom must be true")
    if classes != {"litoria_aurea": 1, "non_target": 0}:
        raise ConfigError("classes must match the canonical mapping")

    return PreprocessingConfig(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        audio=audio,
        spectrogram=spectrogram,
        normalization=normalization,
        rendering=rendering,
        classes=MappingProxyType(classes),
        source_path=path.resolve(),
        sha256=hashlib.sha256(source).hexdigest(),
    )
