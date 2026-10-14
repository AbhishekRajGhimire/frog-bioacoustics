from __future__ import annotations


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
