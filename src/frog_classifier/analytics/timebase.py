"""The analysis time basis: one fixed UTC offset for every window.

Frogs do not follow daylight saving, so times are never converted to a wall-clock
time zone, which would shift activity by an hour at every change.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable


_OFFSET = re.compile(r"(?P<sign>[+-])(?P<hours>\d{2}):(?P<minutes>\d{2})")
# Every UTC offset in use lies between -12:00 and +14:00.
_LARGEST_OFFSET = timedelta(hours=14)


class OffsetError(ValueError):
    pass


@dataclass(frozen=True)
class TimeBasis:
    zone: timezone
    offset: str | None

    @property
    def label(self) -> str:
        if self.offset is None:
            return "recorder clock, UTC offset unknown"
        return f"UTC{self.offset}"


def parse_offset(text: str) -> timezone:
    """A fixed time zone from text such as +11:00 or -03:30."""
    match = _OFFSET.fullmatch(text)
    if match is None:
        raise OffsetError(f"a UTC offset looks like +11:00 or -03:30, not {text!r}")
    size = timedelta(hours=int(match["hours"]), minutes=int(match["minutes"]))
    if int(match["minutes"]) >= 60 or size > _LARGEST_OFFSET:
        raise OffsetError(f"{text!r} is not a real UTC offset")
    return timezone(size if match["sign"] == "+" else -size)


def format_offset(zone: timezone) -> str:
    """Text such as +11:00 for a fixed time zone."""
    offset = zone.utcoffset(None)
    sign = "-" if offset < timedelta(0) else "+"
    minutes = abs(int(offset.total_seconds())) // 60
    return f"{sign}{minutes // 60:02d}:{minutes % 60:02d}"


def most_common_offset(offsets: Iterable[str | None]) -> str | None:
    """The offset most recordings state, ties going to the first in sorted order."""
    counts = Counter(offset for offset in offsets if offset is not None)
    if not counts:
        return None
    return min(counts, key=lambda offset: (-counts[offset], offset))


def choose_time_basis(requested: str | None, recording_offsets: Iterable[str | None]) -> TimeBasis:
    """The requested offset, or else the one most recordings state.

    When no recording states an offset, times stay on the recorders' own clock.
    """
    offset = requested if requested is not None else most_common_offset(recording_offsets)
    if offset is None:
        return TimeBasis(timezone.utc, None)
    zone = parse_offset(offset)
    return TimeBasis(zone, format_offset(zone))


def window_time(started_at: str, recording_offset: str | None, start_s: int, analysis_zone: timezone) -> datetime:
    """When a window starts, in the analysis time zone.

    A recording without a stated offset is taken to use the analysis offset.
    """
    recording_zone = analysis_zone if recording_offset is None else parse_offset(recording_offset)
    start = datetime.fromisoformat(started_at).replace(tzinfo=recording_zone) + timedelta(seconds=start_s)
    return start.astimezone(analysis_zone)
