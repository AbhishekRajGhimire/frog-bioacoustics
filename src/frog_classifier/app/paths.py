"""Where the app keeps models, uploaded recordings, and detection records."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_VARIABLE = "FROG_CLASSIFIER_ROOT"


@dataclass(frozen=True)
class AppPaths:
    root: Path

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def uploads(self) -> Path:
        return self.root / "raw" / "uploads"

    @property
    def detections(self) -> Path:
        return self.root / "results" / "detections"


def app_paths(default_root: Path) -> AppPaths:
    """The data root named by FROG_CLASSIFIER_ROOT, or else default_root."""
    configured = os.environ.get(ROOT_VARIABLE)
    return AppPaths(Path(configured) if configured else default_root)
