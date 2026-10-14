from __future__ import annotations

"""
Baseline training on labeled spectrogram PNGs.

This is intentionally simple and dependency-light:
  - loads and validates labeled/manifest.csv (from build_manifest.py)
  - converts PNGs -> small grayscale arrays
  - trains a Logistic Regression classifier (scikit-learn)
  - reports metrics on train, validation, and test splits

This is a sanity-check baseline. Once you have enough labels, you’ll likely move
to a CNN (PyTorch) for better performance.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from frog_classifier.data.config import load_preprocessing_config
from frog_classifier.data.manifest import load_manifest
from frog_classifier.training.baseline import baseline_features, fit_baseline


@dataclass(frozen=True)
class Example:
    path: Path
    y: int
    split: str


def featurize(paths: Iterable[Path], *, size: int) -> np.ndarray:
    return baseline_features(_grayscale_images(paths), size=size)


def _grayscale_images(paths: Iterable[Path]) -> Iterable[Image.Image]:
    for path in paths:
        with Image.open(path) as image:
            yield image.convert("L")


def eval_split(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"\n== {name} ==")
    print("accuracy:", float(accuracy_score(y_true, y_pred)))
    print("confusion_matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_manifest = repo_root / "labeled" / "manifest.csv"

    p = argparse.ArgumentParser(description="Train a simple baseline on spectrogram PNGs.")
    p.add_argument("--manifest", type=Path, default=default_manifest)
    p.add_argument("--size", type=int, default=64, help="Resize images to NxN before flattening.")
    p.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for LogisticRegression.")
    args = p.parse_args()

    manifest = args.manifest if args.manifest.is_absolute() else (repo_root / args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    config = load_preprocessing_config(repo_root / "config" / "preprocessing.toml")
    rows = load_manifest(
        manifest,
        repo_root,
        config,
        label_root=repo_root / "labeled",
    )
    examples = [
        Example(repo_root / Path(row.image_path), row.label, row.split)
        for row in rows
    ]

    train = [e for e in examples if e.split == "train"]
    val = [e for e in examples if e.split == "val"]
    test = [e for e in examples if e.split == "test"]

    y_train = np.array([e.y for e in train], dtype=np.int64)
    if len(set(y_train.tolist())) < 2:
        print("Not enough label diversity to train (need at least one frog and one background in train).")
        print(f"Train size: {len(train)} | labels: {sorted(set(y_train.tolist()))}")
        return 0

    X_train = featurize([e.path for e in train], size=int(args.size))

    clf = fit_baseline(X_train, y_train, inverse_regularization=float(args.C))

    # Train metrics (sanity check)
    eval_split("train", y_train, clf.predict(X_train))

    X_val = featurize([e.path for e in val], size=int(args.size))
    y_val = np.array([e.y for e in val], dtype=np.int64)
    eval_split("val", y_val, clf.predict(X_val))

    X_test = featurize([e.path for e in test], size=int(args.size))
    y_test = np.array([e.y for e in test], dtype=np.int64)
    eval_split("test", y_test, clf.predict(X_test))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
