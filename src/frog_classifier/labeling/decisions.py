from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from frog_classifier.data.manifest import HOLDING_DIRECTORIES
from frog_classifier.data.naming import ExampleNameError, parse_example_filename


CLASS_FOLDERS = ("litoria_aurea", "non_target")
HOLDING_FOLDER = HOLDING_DIRECTORIES[0]
DECISION_FOLDERS = (*CLASS_FOLDERS, HOLDING_FOLDER)
LOG_COLUMNS = ("example_id", "decision", "action", "faint", "decided_at")
LOG_NAME = "decisions.csv"


@dataclass(frozen=True)
class DecisionRow:
    example_id: str
    decision: str
    action: str
    faint: bool | None
    decided_at: str


@dataclass(frozen=True)
class Move:
    source: Path
    destination: Path


@dataclass(frozen=True)
class ClassProgress:
    clips: int
    recordings: int


class DecisionLog:
    """Append-only record of every labeling decision. Nothing reads it to
    build training data; it exists so a changed mind leaves a trace."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, example_id: str, decision: str, action: str, *, faint: bool | None = None) -> DecisionRow:
        row = DecisionRow(example_id, decision, action, faint, _utc_now())
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        faint_cell = "1" if row.faint is True else "0" if row.faint is False else ""
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if is_new:
                writer.writerow(LOG_COLUMNS)
            writer.writerow([row.example_id, row.decision, row.action, faint_cell, row.decided_at])
        return row

    def rows(self) -> list[DecisionRow]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            return [
                DecisionRow(
                    record["example_id"],
                    record["decision"],
                    record["action"],
                    {"1": True, "0": False}.get(record["faint"]),
                    record["decided_at"],
                )
                for record in csv.DictReader(handle)
            ]


def current_folder(image_path: Path, labeled_root: Path) -> str | None:
    """The decision folder holding the clip, or None when it is outside them."""
    try:
        relative = image_path.resolve().relative_to(labeled_root.resolve())
    except ValueError:
        return None
    if len(relative.parts) == 2 and relative.parts[0] in DECISION_FOLDERS:
        return relative.parts[0]
    return None


def record_decision(
    image_path: Path,
    decision: str,
    *,
    labeled_root: Path,
    log: DecisionLog,
    faint: bool = False,
    chunk_seconds: int = 5,
) -> Move:
    """Move the clip into the decision folder and log it. Never overwrites."""
    if decision not in DECISION_FOLDERS:
        raise ValueError(f"decision must be one of {DECISION_FOLDERS}")
    if faint and decision != "litoria_aurea":
        raise ValueError("faint applies to litoria_aurea decisions only")
    key = parse_example_filename(image_path, chunk_seconds=chunk_seconds)
    origin = current_folder(image_path, labeled_root)
    if origin == decision:
        raise ValueError("the clip is already in that folder; confirm it instead")
    destination = labeled_root / decision / image_path.name
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(image_path, destination)
    try:
        log.append(key.example_id, decision, "label" if origin is None else "change", faint=faint)
    except OSError:
        # Keep the folders and the log consistent: a move without a row is
        # invisible to the audit, so put the clip back before failing.
        os.replace(destination, image_path)
        raise
    return Move(source=image_path, destination=destination)


def confirm_decision(
    image_path: Path,
    *,
    labeled_root: Path,
    log: DecisionLog,
    faint: bool | None = None,
    chunk_seconds: int = 5,
) -> DecisionRow:
    """Log that an audited clip keeps its current folder. Moves nothing."""
    key = parse_example_filename(image_path, chunk_seconds=chunk_seconds)
    origin = current_folder(image_path, labeled_root)
    if origin is None:
        raise ValueError("only a clip inside the labeled folders can be confirmed")
    if faint and origin != "litoria_aurea":
        raise ValueError("faint applies to litoria_aurea decisions only")
    return log.append(key.example_id, origin, "confirm", faint=faint)


def undo_move(
    move: Move,
    *,
    labeled_root: Path,
    log: DecisionLog,
    chunk_seconds: int = 5,
) -> Path:
    """Return the clip to where it came from and log where it went."""
    if not move.destination.exists():
        raise FileNotFoundError(f"nothing to undo at {move.destination}")
    if move.source.exists():
        raise FileExistsError(f"refusing to overwrite {move.source}")
    move.source.parent.mkdir(parents=True, exist_ok=True)
    os.replace(move.destination, move.source)
    key = parse_example_filename(move.source, chunk_seconds=chunk_seconds)
    log.append(key.example_id, current_folder(move.source, labeled_root) or "queue", "undo", faint=None)
    return move.source


def summarize_labels(labeled_root: Path, *, chunk_seconds: int = 5) -> dict[str, ClassProgress]:
    """Clip and distinct-recording counts per decision folder."""
    summary: dict[str, ClassProgress] = {}
    for folder in DECISION_FOLDERS:
        directory = labeled_root / folder
        paths = (
            sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.casefold() == ".png")
            if directory.is_dir()
            else []
        )
        recordings: set[str] = set()
        for path in paths:
            try:
                recordings.add(parse_example_filename(path, chunk_seconds=chunk_seconds).recording_id)
            except ExampleNameError:
                continue
        summary[folder] = ClassProgress(clips=len(paths), recordings=len(recordings))
    return summary


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
