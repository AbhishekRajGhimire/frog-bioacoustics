from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExampleKey:
    recording_id: str
    start_s: int

    @property
    def example_id(self) -> str:
        return f"{self.recording_id}_start{self.start_s}s"


class ExampleNameError(ValueError):
    pass


def parse_example_filename(path: Path, *, chunk_seconds: int) -> ExampleKey:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    match = re.fullmatch(
        r"(?P<recording_id>.+)_start(?P<start_s>\d+)s\.png",
        path.name,
    )
    if match is None:
        raise ExampleNameError(f"invalid example filename: {path.name}")
    start_s = int(match.group("start_s"))
    if start_s % chunk_seconds:
        raise ExampleNameError(
            f"start time {start_s} is not aligned to {chunk_seconds}-second chunks"
        )
    return ExampleKey(recording_id=match.group("recording_id"), start_s=start_s)
