"""The saved model the app scans with, found without loading PyTorch."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from frog_classifier.training.artifact import RUN_METADATA, read_run


# Run folders are named <architecture>-<UTC training time>, such as resnet18-20261014T114432Z.
_TRAINED_AT = re.compile(r"-(\d{8}T\d{6}Z)$")


class ModelError(ValueError):
    pass


@dataclass(frozen=True)
class ModelInfo:
    run_dir: Path
    name: str
    threshold: float


def newest_model(models_root: Path) -> ModelInfo:
    """The most recently trained saved run under models_root.

    Only run.json is read here; the scan process checks every file of the run
    before it uses the model.
    """
    candidates = []
    for run_dir in sorted(models_root.iterdir()) if models_root.is_dir() else ():
        trained_at = _TRAINED_AT.search(run_dir.name)
        if trained_at is None or not (run_dir / RUN_METADATA).is_file():
            continue
        try:
            metadata = read_run(run_dir)
            model = ModelInfo(run_dir, str(metadata["run_name"]), float(metadata["threshold"]["value"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        candidates.append((trained_at.group(1), model))
    if not candidates:
        raise ModelError(f"there is no saved model in {models_root}; train one with scripts/train_model.py")
    return max(candidates, key=lambda candidate: candidate[0])[1]
