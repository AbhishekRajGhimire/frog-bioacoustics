"""The training contract in config/training.toml, loaded and validated."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from frog_classifier.data.config import ConfigError, require_section, require_value


SUPPORTED_SCHEMA_VERSION = 1
_TOP_LEVEL_KEYS = {"schema_version", "seed", "model", "optimization", "augmentation", "baseline"}


@dataclass(frozen=True)
class OptimizationConfig:
    batch_size: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int


@dataclass(frozen=True)
class AugmentationConfig:
    time_mask_max_frames: int
    frequency_mask_max_bins: int
    max_attenuation_db: float


@dataclass(frozen=True)
class BaselineConfig:
    image_size: int
    inverse_regularization: float


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    pretrained: bool
    optimization: OptimizationConfig
    augmentation: AugmentationConfig
    baseline: BaselineConfig
    source_path: Path
    sha256: str


def load_training_config(path: Path) -> TrainingConfig:
    source = path.read_bytes()
    try:
        data = tomllib.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"invalid training TOML: {error}") from error

    if "schema_version" in data and require_value(data, "schema_version", int) != SUPPORTED_SCHEMA_VERSION:
        raise ConfigError(f"training schema_version must be {SUPPORTED_SCHEMA_VERSION}")
    if set(data) != _TOP_LEVEL_KEYS:
        raise ConfigError(
            f"invalid training keys: {sorted(data)}; "
            f"missing: {sorted(_TOP_LEVEL_KEYS - set(data))}; "
            f"unexpected: {sorted(set(data) - _TOP_LEVEL_KEYS)}"
        )

    model_data = require_section(data, "model", {"pretrained"})
    optimization_data = require_section(data, "optimization", {
        "batch_size", "max_epochs", "learning_rate", "weight_decay", "early_stopping_patience",
    })
    augmentation_data = require_section(data, "augmentation", {
        "time_mask_max_frames", "frequency_mask_max_bins", "max_attenuation_db",
    })
    baseline_data = require_section(data, "baseline", {"image_size", "inverse_regularization"})

    seed = int(require_value(data, "seed", int))
    pretrained = bool(require_value(model_data, "pretrained", bool))
    optimization = OptimizationConfig(
        batch_size=int(require_value(optimization_data, "batch_size", int)),
        max_epochs=int(require_value(optimization_data, "max_epochs", int)),
        learning_rate=float(require_value(optimization_data, "learning_rate", float)),
        weight_decay=float(require_value(optimization_data, "weight_decay", float)),
        early_stopping_patience=int(require_value(optimization_data, "early_stopping_patience", int)),
    )
    augmentation = AugmentationConfig(
        time_mask_max_frames=int(require_value(augmentation_data, "time_mask_max_frames", int)),
        frequency_mask_max_bins=int(require_value(augmentation_data, "frequency_mask_max_bins", int)),
        max_attenuation_db=float(require_value(augmentation_data, "max_attenuation_db", float)),
    )
    baseline = BaselineConfig(
        image_size=int(require_value(baseline_data, "image_size", int)),
        inverse_regularization=float(require_value(baseline_data, "inverse_regularization", float)),
    )

    if seed < 0:
        raise ConfigError("seed must be nonnegative")
    if min(optimization.batch_size, optimization.max_epochs, optimization.early_stopping_patience) < 1:
        raise ConfigError("batch_size, max_epochs, and early_stopping_patience must be at least 1")
    if optimization.learning_rate <= 0 or optimization.weight_decay < 0:
        raise ConfigError("learning_rate must be positive and weight_decay nonnegative")
    if (
        min(augmentation.time_mask_max_frames, augmentation.frequency_mask_max_bins) < 0
        or augmentation.max_attenuation_db < 0
    ):
        raise ConfigError("augmentation limits must be nonnegative")
    if baseline.image_size < 1 or baseline.inverse_regularization <= 0:
        raise ConfigError("baseline image_size must be at least 1 and inverse_regularization positive")

    return TrainingConfig(
        seed=seed,
        pretrained=pretrained,
        optimization=optimization,
        augmentation=augmentation,
        baseline=baseline,
        source_path=path.resolve(),
        sha256=hashlib.sha256(source).hexdigest(),
    )
