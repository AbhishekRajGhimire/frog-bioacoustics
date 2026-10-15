"""Where the app keeps models, uploaded recordings, and detection records."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_VARIABLE = "FROG_CLASSIFIER_ROOT"
MODELS_VARIABLE = "FROG_CLASSIFIER_MODELS"


@dataclass(frozen=True)
class AppPaths:
    root: Path
    # The site staff package keeps its model beside the app and its data elsewhere.
    model_folder: Path | None = None

    @property
    def models(self) -> Path:
        return self.model_folder if self.model_folder is not None else self.root / "models"

    @property
    def uploads(self) -> Path:
        return self.root / "raw" / "uploads"

    @property
    def detections(self) -> Path:
        return self.root / "results" / "detections"


def app_paths(default_root: Path) -> AppPaths:
    """The data root named by FROG_CLASSIFIER_ROOT, or else default_root, and the
    model folder named by FROG_CLASSIFIER_MODELS, or else the root's models folder."""
    root = os.environ.get(ROOT_VARIABLE)
    models = os.environ.get(MODELS_VARIABLE)
    return AppPaths(Path(root) if root else default_root, Path(models) if models else None)
