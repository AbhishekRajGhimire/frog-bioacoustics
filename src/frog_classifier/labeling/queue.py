from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from frog_classifier.data.naming import ExampleNameError, parse_example_filename


NIGHT_HOURS = frozenset(range(19, 24)) | frozenset(range(0, 7))

# A recording ID such as 20220121_030000 or 2MM03935_20251130_180000
# carries the start time as YYYYMMDD_HHMMSS.
_TIMESTAMP = re.compile(r"(?<!\d)\d{8}_(\d{2})\d{4}(?!\d)")


def hour_of_day(recording_id: str) -> int | None:
    """Hour from the recording ID's timestamp, or None when it has none."""
    match = _TIMESTAMP.search(recording_id)
    if match is None:
        return None
    hour = int(match.group(1))
    return hour if hour < 24 else None


def is_night(recording_id: str) -> bool:
    hour = hour_of_day(recording_id)
    return hour is not None and hour in NIGHT_HOURS


def build_queue(
    image_paths: Iterable[Path],
    *,
    seed: int,
    chunk_seconds: int = 5,
    per_recording_cap: int = 5,
    daytime_share: float = 0.1,
) -> list[Path]:
    """Session order: shuffled night clips, capped per recording, with a
    share of daytime clips interleaved. Files that do not follow the naming
    contract are left out because they could never be labeled."""
    if per_recording_cap < 1:
        raise ValueError("per_recording_cap must be at least 1")
    if not 0.0 <= daytime_share < 1.0:
        raise ValueError("daytime_share must be at least 0 and below 1")

    night: list[tuple[str, Path]] = []
    day: list[tuple[str, Path]] = []
    for path in sorted(image_paths, key=lambda candidate: candidate.as_posix()):
        try:
            key = parse_example_filename(path, chunk_seconds=chunk_seconds)
        except ExampleNameError:
            continue
        (night if is_night(key.recording_id) else day).append((key.recording_id, path))

    rng = random.Random(seed)
    return _interleave(
        _cap(_shuffled(night, rng), per_recording_cap),
        _cap(_shuffled(day, rng), per_recording_cap),
        daytime_share,
    )


def _shuffled(items: Sequence[tuple[str, Path]], rng: random.Random) -> list[tuple[str, Path]]:
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled


def _cap(items: Sequence[tuple[str, Path]], cap: int) -> list[Path]:
    seen: dict[str, int] = defaultdict(int)
    kept: list[Path] = []
    for recording_id, path in items:
        if seen[recording_id] < cap:
            seen[recording_id] += 1
            kept.append(path)
    return kept


def _interleave(night: list[Path], day: list[Path], daytime_share: float) -> list[Path]:
    if daytime_share == 0.0:
        return night + day
    run = max(1, round((1.0 - daytime_share) / daytime_share))
    result: list[Path] = []
    next_night = 0
    next_day = 0
    while next_night < len(night) or next_day < len(day):
        for _ in range(run):
            if next_night < len(night):
                result.append(night[next_night])
                next_night += 1
        if next_day < len(day):
            result.append(day[next_day])
            next_day += 1
    return result
