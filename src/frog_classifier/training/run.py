"""One complete training run, from the validated manifest to a saved run folder."""

from __future__ import annotations

import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torchvision
from PIL import Image

from frog_classifier.data.config import PreprocessingConfig, load_preprocessing_config
from frog_classifier.data.manifest import ManifestRow, load_manifest
from frog_classifier.data.naming import parse_example_filename
from frog_classifier.labeling.decisions import LOG_NAME, DecisionLog, latest_decisions

from .artifact import (
    ARTIFACT_SCHEMA_VERSION,
    MODEL_FILE,
    PREDICTION_COLUMNS,
    PREDICTIONS_FILE,
    file_sha256,
    finish_run,
    start_run,
    write_csv,
)
from .baseline import baseline_features, fit_baseline
from .config import TrainingConfig, load_training_config
from .fit import FitResult, fit_model, predict_probabilities, select_device
from .images import expected_image_shape, load_spectrograms
from .metrics import (
    THRESHOLD_RULE,
    counts_by_hour,
    evaluate,
    false_detections_per_hour,
    recall_by_faint,
    select_threshold,
)
from .model import ARCHITECTURE, PRETRAINED_WEIGHTS
from .report import CurveInput, render_report, write_precision_recall_figure


HISTORY_COLUMNS = ("epoch", "train_loss", "val_loss", "val_average_precision", "learning_rate", "seconds")
CROSS_VALIDATION_COLUMNS = (
    "rotation", "test_fold", "validation_fold", "best_epoch", "model", "threshold",
    "average_precision", "precision", "recall", "f1", "false_positive_rate",
)
OUT_OF_FOLD_COLUMNS = ("example_id", "image_path", "recording_id", "fold", "label", "faint", "probability", "threshold")


@dataclass(frozen=True)
class RunPaths:
    repo_root: Path
    manifest: Path
    preprocessing_config: Path
    training_config: Path
    labeled_root: Path
    models_root: Path

    @classmethod
    def defaults(cls, repo_root: Path) -> RunPaths:
        return cls(
            repo_root=repo_root,
            manifest=repo_root / "labeled" / "manifest.csv",
            preprocessing_config=repo_root / "config" / "preprocessing.toml",
            training_config=repo_root / "config" / "training.toml",
            labeled_root=repo_root / "labeled",
            models_root=repo_root / "models",
        )


@dataclass(frozen=True)
class _Dataset:
    rows: tuple[ManifestRow, ...]
    pixels: np.ndarray
    sha256: tuple[str, ...]
    labels: np.ndarray
    folds: np.ndarray
    faint: np.ndarray
    baseline_features: np.ndarray


@dataclass(frozen=True)
class _Rotation:
    test_fold: int
    val_fold: int
    fit: FitResult
    resnet: np.ndarray
    baseline: np.ndarray
    resnet_threshold: float
    baseline_threshold: float


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def train_and_save(
    paths: RunPaths,
    *,
    cross_validate: bool = True,
    device: torch.device | None = None,
    progress: Callable[[str], None] = print,
    clock: Callable[[], datetime] = utc_now,
) -> Path:
    """Train, evaluate, and save one run; return its folder.

    Rotation zero uses the manifest's own test and validation folds and is the
    saved model. Cross-validation adds one rotation per remaining fold.
    """
    preprocessing = load_preprocessing_config(paths.preprocessing_config)
    training = load_training_config(paths.training_config)
    rows = load_manifest(paths.manifest, paths.repo_root, preprocessing, label_root=paths.labeled_root)
    test_fold, val_fold = _manifest_folds(rows)
    created_at = clock()
    name = f"{ARCHITECTURE}-{created_at:%Y%m%dT%H%M%SZ}"
    staging = start_run(paths.models_root, name)
    device = device if device is not None else select_device()

    progress(f"Loading {len(rows)} images")
    data = _load_dataset(rows, paths, preprocessing, training)
    fold_ids = sorted(set(data.folds.tolist()))
    count = len(fold_ids) if cross_validate else 1
    rotations: list[_Rotation] = []
    started = time.perf_counter()
    for offset in range(count):
        rotation_test = fold_ids[(fold_ids.index(test_fold) + offset) % len(fold_ids)]
        rotation_val = fold_ids[(fold_ids.index(val_fold) + offset) % len(fold_ids)]
        progress(f"Rotation {offset + 1} of {count}: test fold {rotation_test}, validation fold {rotation_val}, device {device}")
        rotations.append(_train_rotation(data, rotation_test, rotation_val, training, preprocessing, device, progress))
    seconds = time.perf_counter() - started

    metadata = _metadata(
        name, created_at, data, rotations, training, preprocessing, paths, device, seconds, cross_validate,
    )
    _write_files(staging, data, rotations, training, preprocessing, paths, metadata, cross_validate)
    return finish_run(staging, metadata)


def _manifest_folds(rows: tuple[ManifestRow, ...]) -> tuple[int, int]:
    """The single test fold and single validation fold the manifest's split column names."""
    folds_by_split: dict[str, set[int]] = {}
    for row in rows:
        folds_by_split.setdefault(row.split, set()).add(row.fold)
    test, val = folds_by_split.get("test", set()), folds_by_split.get("val", set())
    train = folds_by_split.get("train", set())
    if len(test) != 1 or len(val) != 1 or test & train or val & train:
        raise ValueError("the manifest must hold exactly one test fold and one validation fold, apart from training folds")
    return next(iter(test)), next(iter(val))


def _load_dataset(
    rows: tuple[ManifestRow, ...],
    paths: RunPaths,
    preprocessing: PreprocessingConfig,
    training: TrainingConfig,
) -> _Dataset:
    loaded = load_spectrograms([paths.repo_root / row.image_path for row in rows], expected_image_shape(preprocessing))
    faint_ids = _faint_example_ids(paths.labeled_root, preprocessing.audio.chunk_seconds)
    return _Dataset(
        rows=rows,
        pixels=loaded.pixels,
        sha256=loaded.sha256,
        labels=np.array([row.label for row in rows], dtype=np.int64),
        folds=np.array([row.fold for row in rows], dtype=np.int64),
        faint=np.array([row.label == 1 and row.example_id in faint_ids for row in rows], dtype=bool),
        baseline_features=baseline_features(
            (Image.fromarray(image) for image in loaded.pixels), size=training.baseline.image_size,
        ),
    )


def _faint_example_ids(labeled_root: Path, chunk_seconds: int) -> frozenset[str]:
    """Frog clips whose latest frog decision used the Frog faint button, per the decision log."""
    clips = latest_decisions(labeled_root, DecisionLog(labeled_root / LOG_NAME), chunk_seconds=chunk_seconds)
    return frozenset(
        parse_example_filename(clip.path, chunk_seconds=chunk_seconds).example_id
        for clip in clips
        if clip.folder == "litoria_aurea" and clip.faint
    )


def _train_rotation(
    data: _Dataset,
    test_fold: int,
    val_fold: int,
    training: TrainingConfig,
    preprocessing: PreprocessingConfig,
    device: torch.device,
    progress: Callable[[str], None],
) -> _Rotation:
    train = (data.folds != test_fold) & (data.folds != val_fold)
    val = data.folds == val_fold
    fit = fit_model(
        data.pixels[train], data.labels[train], data.pixels[val], data.labels[val], training,
        db_floor=preprocessing.normalization.db_floor,
        db_ceiling=preprocessing.normalization.db_ceiling,
        device=device,
        progress=progress,
    )
    resnet = predict_probabilities(fit.model, data.pixels, device)
    classifier = fit_baseline(
        data.baseline_features[train], data.labels[train],
        inverse_regularization=training.baseline.inverse_regularization,
    )
    baseline = classifier.predict_proba(data.baseline_features)[:, list(classifier.classes_).index(1)]
    return _Rotation(
        test_fold=test_fold,
        val_fold=val_fold,
        fit=fit,
        resnet=resnet,
        baseline=baseline,
        resnet_threshold=select_threshold(data.labels[val], resnet[val]),
        baseline_threshold=select_threshold(data.labels[val], baseline[val]),
    )


def _metadata(
    name: str,
    created_at: datetime,
    data: _Dataset,
    rotations: list[_Rotation],
    training: TrainingConfig,
    preprocessing: PreprocessingConfig,
    paths: RunPaths,
    device: torch.device,
    seconds: float,
    cross_validate: bool,
) -> dict[str, Any]:
    primary = rotations[0]
    labels = data.labels
    masks = {
        "train": (data.folds != primary.test_fold) & (data.folds != primary.val_fold),
        "validation": data.folds == primary.val_fold,
        "test": data.folds == primary.test_fold,
    }
    test, val = masks["test"], masks["validation"]
    resnet_test = evaluate(labels[test], primary.resnet[test], primary.resnet_threshold)
    recording_ids = np.array([row.recording_id for row in data.rows])
    uv_lock = paths.repo_root / "uv.lock"
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_name": name,
        "created_at": created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "architecture": ARCHITECTURE,
            "pretrained_weights": PRETRAINED_WEIGHTS if training.pretrained else None,
            "input_shape": [1, *data.pixels.shape[1:]],
            "classes": dict(preprocessing.classes),
            "positive_class": "litoria_aurea",
            "output": "one logit per image; the frog probability is its sigmoid",
        },
        "threshold": {"value": primary.resnet_threshold, "rule": THRESHOLD_RULE},
        "seed": training.seed,
        "training": {
            "best_epoch": primary.fit.best_epoch,
            "epochs_run": len(primary.fit.history),
            "device": device.type,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "seconds": round(seconds, 1),
        },
        "data": {
            "examples": len(data.rows),
            "test_fold": primary.test_fold,
            "validation_fold": primary.val_fold,
            "faint_tagged_frog_clips": int(data.faint.sum()),
            "splits": {
                split: {
                    "examples": int(mask.sum()),
                    "frog": int(labels[mask].sum()),
                    "background": int((labels[mask] == 0).sum()),
                    "recordings": len(set(recording_ids[mask].tolist())),
                }
                for split, mask in masks.items()
            },
        },
        "checksums": {
            "manifest": file_sha256(paths.manifest),
            "preprocessing_config": preprocessing.sha256,
            "training_config": training.sha256,
            "uv_lock": file_sha256(uv_lock) if uv_lock.is_file() else None,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "platform": platform.platform(),
        },
        "source": _git_state(paths.repo_root),
        "metrics": {
            "resnet": {
                "validation": asdict(evaluate(labels[val], primary.resnet[val], primary.resnet_threshold)),
                "test": asdict(resnet_test),
            },
            "baseline": {
                "threshold": primary.baseline_threshold,
                "validation": asdict(evaluate(labels[val], primary.baseline[val], primary.baseline_threshold)),
                "test": asdict(evaluate(labels[test], primary.baseline[test], primary.baseline_threshold)),
            },
            "false_detections_per_hour": false_detections_per_hour(
                resnet_test.false_positive_rate, preprocessing.audio.chunk_seconds,
            ),
            "test_by_hour": [
                asdict(row)
                for row in counts_by_hour(
                    recording_ids[test].tolist(), labels[test], primary.resnet[test], primary.resnet_threshold,
                )
            ],
            "faint_recall": {
                split: {
                    group: asdict(count)
                    for group, count in recall_by_faint(
                        labels[masks[split]], data.faint[masks[split]], primary.resnet[masks[split]],
                        primary.resnet_threshold,
                    ).items()
                }
                for split in ("validation", "test")
            },
        },
        "cross_validation": _cross_validation(data, rotations) if cross_validate else None,
    }


def _cross_validation(data: _Dataset, rotations: list[_Rotation]) -> dict[str, Any]:
    per_rotation = []
    for index, rotation in enumerate(rotations):
        test = data.folds == rotation.test_fold
        per_rotation.append({
            "rotation": index,
            "test_fold": rotation.test_fold,
            "validation_fold": rotation.val_fold,
            "best_epoch": rotation.fit.best_epoch,
            "resnet_threshold": rotation.resnet_threshold,
            "baseline_threshold": rotation.baseline_threshold,
            "resnet": asdict(evaluate(data.labels[test], rotation.resnet[test], rotation.resnet_threshold)),
            "baseline": asdict(evaluate(data.labels[test], rotation.baseline[test], rotation.baseline_threshold)),
        })
    summary = {
        model: {
            metric: _spread([row[model][metric] for row in per_rotation])
            for metric in ("average_precision", "precision", "recall", "f1")
        }
        for model in ("resnet", "baseline")
    }
    return {"rotations": per_rotation, "summary": summary}


def _spread(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _git_state(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_worktree_dirty": None}
    return {"git_commit": commit, "git_worktree_dirty": bool(status.strip())}


def _write_files(
    staging: Path,
    data: _Dataset,
    rotations: list[_Rotation],
    training: TrainingConfig,
    preprocessing: PreprocessingConfig,
    paths: RunPaths,
    metadata: dict[str, Any],
    cross_validate: bool,
) -> None:
    primary = rotations[0]
    torch.save({key: value.cpu() for key, value in primary.fit.model.state_dict().items()}, staging / MODEL_FILE)
    shutil.copyfile(preprocessing.source_path, staging / "preprocessing.toml")
    shutil.copyfile(training.source_path, staging / "training.toml")
    shutil.copyfile(paths.manifest, staging / "manifest.csv")
    write_csv(staging / PREDICTIONS_FILE, PREDICTION_COLUMNS, (
        {
            "example_id": row.example_id,
            "image_path": row.image_path,
            "image_sha256": checksum,
            "recording_id": row.recording_id,
            "fold": row.fold,
            "split": row.split,
            "label": row.label,
            "faint": int(faint) if row.label == 1 else "",
            "probability": f"{resnet:.8f}",
            "baseline_probability": f"{baseline:.8f}",
        }
        for row, checksum, faint, resnet, baseline in zip(
            data.rows, data.sha256, data.faint, primary.resnet, primary.baseline, strict=True,
        )
    ))
    write_csv(staging / "history.csv", HISTORY_COLUMNS, (asdict(record) for record in primary.fit.history))
    if cross_validate:
        _write_cross_validation(staging, data, rotations, metadata)
    labels = data.labels
    write_precision_recall_figure(staging / "precision_recall.png", [
        CurveInput(
            title, labels[mask], primary.resnet[mask], primary.baseline[mask],
            primary.resnet_threshold, primary.baseline_threshold,
        )
        for title, mask in (
            ("Validation", data.folds == primary.val_fold),
            ("Test", data.folds == primary.test_fold),
        )
    ])
    (staging / "report.md").write_bytes(render_report(metadata).encode("utf-8"))


def _write_cross_validation(
    staging: Path,
    data: _Dataset,
    rotations: list[_Rotation],
    metadata: dict[str, Any],
) -> None:
    write_csv(staging / "cross_validation.csv", CROSS_VALIDATION_COLUMNS, (
        {
            "rotation": row["rotation"],
            "test_fold": row["test_fold"],
            "validation_fold": row["validation_fold"],
            "best_epoch": row["best_epoch"],
            "model": model,
            "threshold": row[f"{model}_threshold"],
            **{
                metric: row[model][metric]
                for metric in ("average_precision", "precision", "recall", "f1", "false_positive_rate")
            },
        }
        for row in metadata["cross_validation"]["rotations"]
        for model in ("resnet", "baseline")
    ))
    by_test_fold = {rotation.test_fold: rotation for rotation in rotations}
    write_csv(staging / "out_of_fold_predictions.csv", OUT_OF_FOLD_COLUMNS, (
        {
            "example_id": row.example_id,
            "image_path": row.image_path,
            "recording_id": row.recording_id,
            "fold": row.fold,
            "label": row.label,
            "faint": int(faint) if row.label == 1 else "",
            "probability": f"{by_test_fold[row.fold].resnet[index]:.8f}",
            "threshold": f"{by_test_fold[row.fold].resnet_threshold:.8f}",
        }
        for index, (row, faint) in enumerate(zip(data.rows, data.faint, strict=True))
    ))
