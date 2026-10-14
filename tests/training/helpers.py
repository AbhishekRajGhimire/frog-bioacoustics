from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from frog_classifier.data.manifest import ManifestRow, serialize_manifest
from frog_classifier.labeling.decisions import LOG_NAME, DecisionLog
from tests.data.helpers import load_test_config

if TYPE_CHECKING:
    from frog_classifier.training.run import RunPaths


IMAGE_SHAPE = (128, 216)

TRAINING_CONFIG = b'''schema_version = 1
seed = 7

[model]
pretrained = false

[optimization]
batch_size = 4
max_epochs = 2
learning_rate = 0.001
weight_decay = 0.0
early_stopping_patience = 1

[augmentation]
time_mask_max_frames = 4
frequency_mask_max_bins = 4
max_attenuation_db = 3.0

[baseline]
image_size = 8
inverse_regularization = 1.0
'''


def _split(frog: int, background: int, true_positives: int, false_positives: int) -> dict:
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / frog
    return {
        "examples": frog + background, "frog": frog, "background": background,
        "true_positives": true_positives, "false_positives": false_positives,
        "true_negatives": background - false_positives, "false_negatives": frog - true_positives,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
        "false_positive_rate": false_positives / background,
        "average_precision": 0.9, "roc_auc": 0.95,
    }


def example_metadata(*, cross_validation: bool) -> dict:
    """The run.json shape that render_report reads, with small hand-made numbers."""
    rotation = {
        "rotation": 0, "test_fold": 0, "validation_fold": 1, "best_epoch": 3,
        "resnet_threshold": 0.6, "baseline_threshold": 0.4,
        "resnet": _split(20, 50, 18, 2), "baseline": _split(20, 50, 12, 9),
    }
    spread = {"mean": 0.9, "std": 0.02, "min": 0.88, "max": 0.92}
    return {
        "artifact_schema_version": 1,
        "run_name": "resnet18-20261014T000000Z",
        "created_at": "2026-10-14T00:00:00Z",
        "model": {"architecture": "resnet18", "pretrained_weights": "IMAGENET1K_V1", "input_shape": [1, 128, 216]},
        "threshold": {"value": 0.6, "rule": "maximum F1 on the validation fold, ties to the higher threshold"},
        "seed": 1337,
        "training": {"best_epoch": 3, "epochs_run": 7, "device": "cpu", "device_name": "CPU", "seconds": 42.0},
        "data": {
            "examples": 210, "test_fold": 0, "validation_fold": 1, "faint_tagged_frog_clips": 4,
            "splits": {
                "train": {"examples": 70, "frog": 20, "background": 50, "recordings": 9},
                "validation": {"examples": 70, "frog": 20, "background": 50, "recordings": 8},
                "test": {"examples": 70, "frog": 20, "background": 50, "recordings": 7},
            },
        },
        "checksums": {"manifest": "a" * 64, "preprocessing_config": "b" * 64, "training_config": "c" * 64, "uv_lock": "d" * 64},
        "environment": {"python": "3.13.3", "torch": "2.14.1", "torchvision": "0.29.1", "numpy": "2.3.5", "cuda": None, "cudnn": None, "platform": "test"},
        "source": {"git_commit": "f" * 40, "git_worktree_dirty": False},
        "metrics": {
            "resnet": {"validation": _split(20, 50, 17, 3), "test": _split(20, 50, 18, 2)},
            "baseline": {"threshold": 0.4, "validation": _split(20, 50, 11, 8), "test": _split(20, 50, 12, 9)},
            "false_detections_per_hour": 28.8,
            "test_by_hour": [
                {"hour": 1, "background": 30, "false_positives": 2, "frog": 15, "missed": 1},
                {"hour": None, "background": 20, "false_positives": 0, "frog": 5, "missed": 1},
            ],
            "faint_recall": {
                "validation": {"faint": {"clips": 2, "detected": 1}, "other": {"clips": 18, "detected": 16}},
                "test": {"faint": {"clips": 2, "detected": 2}, "other": {"clips": 18, "detected": 16}},
            },
        },
        "cross_validation": (
            {
                "rotations": [rotation],
                "summary": {
                    model: {metric: spread for metric in ("average_precision", "precision", "recall", "f1")}
                    for model in ("resnet", "baseline")
                },
            }
            if cross_validation
            else None
        ),
    }


def build_synthetic_repo(
    repo_root: Path,
    *,
    test_fold: int = 0,
    val_fold: int = 1,
    faint_log: bool = True,
) -> RunPaths:
    """Five folds, each with one frog and one background recording of two clips.

    Frog clips carry a bright band where a call would sit; the first frog clip of
    the test fold is tagged faint in the decision log when faint_log is true.
    """
    # Imported here so the configuration helpers above stay usable without the run module.
    from frog_classifier.training.run import RunPaths

    config = load_test_config(repo_root)
    training_path = repo_root / "config" / "training.toml"
    training_path.write_bytes(TRAINING_CONFIG)
    labeled = repo_root / "labeled"
    rng = np.random.default_rng(0)
    rows = []
    for fold in range(5):
        split = "test" if fold == test_fold else "val" if fold == val_fold else "train"
        for label_name, label, hour in (("litoria_aurea", 1, 2), ("non_target", 0, 22)):
            recording_id = f"2022010{fold + 1}_{hour:02d}0000"
            for start_s in (0, 5):
                example_id = f"{recording_id}_start{start_s}s"
                image_path = f"labeled/{label_name}/{example_id}.png"
                pixels = rng.integers(0, 40, IMAGE_SHAPE, dtype=np.uint8)
                if label:
                    pixels[60:70, :] = 220
                (repo_root / image_path).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(repo_root / image_path)
                rows.append(ManifestRow(
                    manifest_version=1,
                    example_id=example_id,
                    image_path=image_path,
                    label=label,
                    label_name=label_name,
                    recording_id=recording_id,
                    start_s=start_s,
                    fold=fold,
                    split=split,
                    preprocessing_config_sha256=config.sha256,
                ))
    manifest = labeled / "manifest.csv"
    manifest.write_bytes(serialize_manifest(tuple(rows)))
    if faint_log:
        faint_clip = next(row for row in rows if row.fold == test_fold and row.label == 1)
        DecisionLog(labeled / LOG_NAME).append(faint_clip.example_id, "litoria_aurea", "label", faint=True)
    return RunPaths(
        repo_root=repo_root,
        manifest=manifest,
        preprocessing_config=repo_root / "config" / "preprocessing.toml",
        training_config=training_path,
        labeled_root=labeled,
        models_root=repo_root / "models",
    )
