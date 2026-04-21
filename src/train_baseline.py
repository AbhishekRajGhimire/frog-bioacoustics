from __future__ import annotations

"""
Baseline training on labeled spectrogram PNGs.

This is intentionally simple and dependency-light:
  - loads Data/training_data/manifest.csv (from build_manifest.py)
  - converts PNGs -> small grayscale arrays
  - trains a Logistic Regression classifier (scikit-learn)
  - reports metrics on val/test splits (if available)

This is a sanity-check baseline. Once you have enough labels, you’ll likely move
to a CNN (PyTorch) for better performance.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass(frozen=True)
class Example:
    path: Path
    y: int
    split: str


def load_manifest(repo_root: Path, manifest_path: Path) -> list[Example]:
    out: list[Example] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            p = repo_root / Path(row["image_path"])
            out.append(Example(path=p, y=int(row["label"]), split=row.get("split", "train")))
    return out


def featurize(paths: Iterable[Path], *, size: int) -> np.ndarray:
    feats: list[np.ndarray] = []
    for p in paths:
        img = Image.open(p).convert("L")  # grayscale
        img = img.resize((size, size))
        x = np.asarray(img, dtype=np.float32) / 255.0
        feats.append(x.reshape(-1))
    return np.stack(feats, axis=0)


def eval_split(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"\n== {name} ==")
    print("accuracy:", float(accuracy_score(y_true, y_pred)))
    print("confusion_matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_manifest = repo_root / "Data" / "training_data" / "manifest.csv"

    p = argparse.ArgumentParser(description="Train a simple baseline on spectrogram PNGs.")
    p.add_argument("--manifest", type=Path, default=default_manifest)
    p.add_argument("--size", type=int, default=64, help="Resize images to NxN before flattening.")
    p.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for LogisticRegression.")
    args = p.parse_args()

    manifest = args.manifest if args.manifest.is_absolute() else (repo_root / args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    examples = load_manifest(repo_root, manifest)
    examples = [e for e in examples if e.path.exists()]
    if not examples:
        print("No examples found (manifest paths missing).")
        return 0

    train = [e for e in examples if e.split == "train"]
    val = [e for e in examples if e.split == "val"]
    test = [e for e in examples if e.split == "test"]

    # Fallback: if split is too tiny, treat everything as train.
    if len(train) < 2:
        train = examples
        val = []
        test = []

    y_train = np.array([e.y for e in train], dtype=np.int64)
    if len(set(y_train.tolist())) < 2:
        print("Not enough label diversity to train (need at least one frog and one background in train).")
        print(f"Train size: {len(train)} | labels: {sorted(set(y_train.tolist()))}")
        return 0

    X_train = featurize([e.path for e in train], size=int(args.size))

    clf = LogisticRegression(
        C=float(args.C),
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,
    )
    clf.fit(X_train, y_train)

    # Train metrics (sanity check)
    eval_split("train", y_train, clf.predict(X_train))

    if val:
        X_val = featurize([e.path for e in val], size=int(args.size))
        y_val = np.array([e.y for e in val], dtype=np.int64)
        eval_split("val", y_val, clf.predict(X_val))
    else:
        print("\n(no val split examples yet)")

    if test:
        X_test = featurize([e.path for e in test], size=int(args.size))
        y_test = np.array([e.y for e in test], dtype=np.int64)
        eval_split("test", y_test, clf.predict(X_test))
    else:
        print("\n(no test split examples yet)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

