from __future__ import annotations

"""
Build a labeled manifest CSV and a leakage-safe split.

We split by recording_id (not by chunk) to avoid leakage: chunks from the same
recording are highly correlated.

Expected labeled folder layout:
  Data/training_data/
    litoria_aurea/*.png
    background/*.png

Expected filename convention:
  <recording_id>_start{N}s.png
Example:
  20220308_050000_start30s.png
  2MM03935_20251130_180000_start230s.png
"""

import argparse
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LABEL_DIRS = {
    "litoria_aurea": 1,
    "non_target": 0,
}

_START_RE = re.compile(r"^(?P<recording_id>.+)_start(?P<sec>\d+)s$", re.IGNORECASE)


@dataclass(frozen=True)
class Row:
    image_path: str
    label: int
    label_name: str
    recording_id: str
    start_s: int
    split: str


def parse_from_filename(p: Path) -> tuple[str, int] | None:
    m = _START_RE.match(p.stem)
    if not m:
        return None
    return m.group("recording_id"), int(m.group("sec"))


def iter_pngs(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.png")


def assign_splits(
    recording_ids: list[str],
    *,
    seed: int,
    val_frac: float,
    test_frac: float,
) -> dict[str, str]:
    if not (0.0 <= val_frac <= 1.0 and 0.0 <= test_frac <= 1.0 and (val_frac + test_frac) < 1.0):
        raise ValueError("val_frac and test_frac must be in [0,1] and sum to < 1.0")

    rnd = random.Random(seed)
    ids = recording_ids[:]
    rnd.shuffle(ids)

    n = len(ids)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test : n_test + n_val])

    out: dict[str, str] = {}
    for rid in ids:
        if rid in test_ids:
            out[rid] = "test"
        elif rid in val_ids:
            out[rid] = "val"
        else:
            out[rid] = "train"
    return out


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_training = repo_root / "labeled"
    default_out = default_training / "manifest.csv"

    p = argparse.ArgumentParser(description="Build manifest.csv from labeled training_data folders.")
    p.add_argument("--training-root", type=Path, default=default_training)
    p.add_argument("--out-csv", type=Path, default=default_out)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    args = p.parse_args()

    training_root: Path = args.training_root if args.training_root.is_absolute() else (repo_root / args.training_root)
    out_csv: Path = args.out_csv if args.out_csv.is_absolute() else (repo_root / args.out_csv)

    rows_no_split: list[tuple[Path, int, str, str, int]] = []
    recording_ids: set[str] = set()

    for label_name, label in LABEL_DIRS.items():
        label_dir = training_root / label_name
        if not label_dir.exists():
            continue
        for img_path in iter_pngs(label_dir):
            parsed = parse_from_filename(img_path)
            if parsed is None:
                continue
            recording_id, start_s = parsed
            rows_no_split.append((img_path, label, label_name, recording_id, start_s))
            recording_ids.add(recording_id)

    if not rows_no_split:
        print(f"No labeled PNGs found under: {training_root}")
        return 0

    split_map = assign_splits(
        sorted(recording_ids),
        seed=int(args.seed),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label", "label_name", "recording_id", "start_s", "split"])
        for img_path, label, label_name, recording_id, start_s in sorted(rows_no_split, key=lambda x: str(x[0])):
            split = split_map[recording_id]
            rel = img_path.relative_to(repo_root)
            row = Row(
                image_path=str(rel).replace("\\", "/"),
                label=int(label),
                label_name=label_name,
                recording_id=recording_id,
                start_s=int(start_s),
                split=split,
            )
            w.writerow([row.image_path, row.label, row.label_name, row.recording_id, row.start_s, row.split])

    print(f"Wrote manifest: {out_csv}")
    print(f"Rows: {len(rows_no_split)} | recordings: {len(recording_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

