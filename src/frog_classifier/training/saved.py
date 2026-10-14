"""Load a saved run for prediction, refusing one that is incomplete or altered."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from frog_classifier.data.config import PreprocessingConfig, load_preprocessing_config

from .artifact import (
    ARTIFACT_SCHEMA_VERSION,
    MODEL_FILE,
    PREPROCESSING_FILE,
    RUN_METADATA,
    check_files,
    read_run,
)
from .model import ARCHITECTURE, SpectrogramResNet


class SavedRunError(ValueError):
    pass


@dataclass(frozen=True)
class SavedRun:
    name: str
    model: SpectrogramResNet
    model_sha256: str
    threshold: float
    preprocessing: PreprocessingConfig


def run_problems(run_dir: Path, metadata: Mapping[str, Any]) -> list[str]:
    """Reasons a run cannot be trusted: a format or architecture this code does
    not know, or a file missing or changed since run.json recorded it."""
    version = metadata.get("artifact_schema_version")
    if version != ARTIFACT_SCHEMA_VERSION:
        return [f"unsupported run format {version!r}"]
    problems = check_files(run_dir, metadata)
    architecture = metadata["model"]["architecture"]
    if architecture != ARCHITECTURE:
        problems.append(f"unsupported architecture {architecture!r}")
    return problems


def load_model(run_dir: Path) -> SpectrogramResNet:
    """The run's trained network on the CPU, in evaluation mode."""
    model = SpectrogramResNet(pretrained=False)
    model.load_state_dict(torch.load(run_dir / MODEL_FILE, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_saved_run(run_dir: Path) -> SavedRun:
    """The run's model and threshold, with the preprocessing contract it was trained on.

    The contract comes from the run's own copy, so a later change to the tracked
    configuration can never change what this model is shown.
    """
    if not (run_dir / RUN_METADATA).is_file():
        raise SavedRunError(f"{run_dir} is not a saved run: it has no {RUN_METADATA}")
    metadata = read_run(run_dir)
    problems = run_problems(run_dir, metadata)
    if not problems:
        problems = [
            f"{name}: not recorded in {RUN_METADATA}"
            for name in (MODEL_FILE, PREPROCESSING_FILE)
            if name not in metadata["files"]
        ]
    if problems:
        raise SavedRunError(f"run {run_dir.name} cannot be used: " + "; ".join(problems))
    return SavedRun(
        name=metadata["run_name"],
        model=load_model(run_dir),
        model_sha256=metadata["files"][MODEL_FILE],
        threshold=float(metadata["threshold"]["value"]),
        preprocessing=load_preprocessing_config(run_dir / PREPROCESSING_FILE),
    )
