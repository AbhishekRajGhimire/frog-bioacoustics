"""Prove that a saved run reproduces its own predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .artifact import MODEL_FILE, PREDICTIONS_FILE, check_files, file_sha256, read_csv, read_run
from .fit import predict_probabilities, use_full_precision
from .images import load_spectrograms
from .model import ARCHITECTURE, SpectrogramResNet


# Saved probabilities have eight decimals; GPU and CPU fp32 agree far more closely than this.
PROBABILITY_TOLERANCE = 1e-4


@dataclass(frozen=True)
class VerificationResult:
    checked: int
    largest_difference: float
    problems: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.problems


def verify_run(run_dir: Path, repo_root: Path, *, device: torch.device) -> VerificationResult:
    """Check the run's file checksums, then recompute every saved probability.

    Images are read from their manifest paths under repo_root. An image that is
    gone or whose bytes changed since training is reported as such and skipped.
    """
    metadata = read_run(run_dir)
    problems = check_files(run_dir, metadata)
    if problems:
        return VerificationResult(0, 0.0, tuple(problems))
    if metadata["model"]["architecture"] != ARCHITECTURE:
        return VerificationResult(0, 0.0, (f"unsupported architecture {metadata['model']['architecture']!r}",))

    use_full_precision()
    model = SpectrogramResNet(pretrained=False)
    model.load_state_dict(torch.load(run_dir / MODEL_FILE, map_location="cpu", weights_only=True))
    model.to(device)

    readable = []
    for row in read_csv(run_dir / PREDICTIONS_FILE):
        path = repo_root / row["image_path"]
        if not path.is_file():
            problems.append(f"{row['example_id']}: image missing at {row['image_path']}")
        elif file_sha256(path) != row["image_sha256"]:
            problems.append(f"{row['example_id']}: image changed since training")
        else:
            readable.append((row, path))
    if not readable:
        return VerificationResult(0, 0.0, tuple(problems))

    shape = tuple(metadata["model"]["input_shape"][1:])
    pixels = load_spectrograms([path for _, path in readable], shape).pixels
    recomputed = predict_probabilities(model, pixels, device)
    saved = np.array([float(row["probability"]) for row, _ in readable])
    differences = np.abs(recomputed - saved)
    for (row, _), new, old, difference in zip(readable, recomputed, saved, differences, strict=True):
        if difference > PROBABILITY_TOLERANCE:
            problems.append(f"{row['example_id']}: probability {new:.6f} differs from saved {old:.6f}")
    return VerificationResult(len(readable), float(differences.max()), tuple(problems))
