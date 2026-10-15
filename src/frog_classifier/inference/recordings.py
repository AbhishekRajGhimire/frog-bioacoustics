"""Find recordings beneath an input folder and read when each one started."""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


AUDIO_SUFFIXES = frozenset({".wav", ".mp3"})
NO_START_TIME = (
    "no start time: the file name has no valid YYYYMMDD_HHMMSS timestamp "
    "and there is no AudioMoth header"
)

_FILE_NAME_TIMESTAMP = re.compile(r"(?<!\d)\d{8}_\d{6}(?!\d)")
# AudioMoth firmware writes, for example,
# "Recorded at 01:00:00 23/01/2022 (UTC+11) by AudioMoth 248D9B0260370525 ...".
_AUDIOMOTH_TIME = re.compile(
    r"Recorded at (?P<clock>\d{2}:\d{2}:\d{2} \d{2}/\d{2}/\d{4}) "
    r"\(UTC(?:(?P<sign>[+-])(?P<hours>\d{1,2})(?::(?P<minutes>\d{2}))?)?\)"
)
# An INFO list holds a few short text fields; anything larger is not worth reading.
_MAX_LIST_BYTES = 1 << 20
# Enough of a file to tell two recordings apart: the recorder's header, with its
# serial number and clock, and the first moments of audio.
_HEAD_BYTES = 64 * 1024


@dataclass(frozen=True)
class RecordingTime:
    started_at: datetime | None
    utc_offset: str | None
    notes: tuple[str, ...]


def find_recordings(input_root: Path) -> list[str]:
    """Every audio file beneath input_root, as sorted POSIX paths relative to it."""
    return sorted(
        path.relative_to(input_root).as_posix()
        for path in input_root.rglob("*")
        if path.suffix.lower() in AUDIO_SUFFIXES and path.is_file()
    )


def file_head_sha256(path: Path) -> str:
    """SHA-256 of a file's first 64 KiB: a cheap check that a stored recording is still the same file."""
    with path.open("rb") as handle:
        return hashlib.sha256(handle.read(_HEAD_BYTES)).hexdigest()


def recording_time(path: Path) -> RecordingTime:
    """The local start time of a recording and its UTC offset when known.

    The file name's YYYYMMDD_HHMMSS timestamp is the start time, as everywhere
    else in the project. An AudioMoth header supplies the UTC offset, and the
    start time when the file name has none.
    """
    named, name_note = _file_name_time(path.stem)
    notes = [name_note] if name_note is not None else []
    header_time, utc_offset = _audiomoth_time(path)
    if named is not None and header_time is not None and named != header_time:
        notes.append(
            f"file name time {named.isoformat()} differs from the AudioMoth header time "
            f"{header_time.isoformat()}; using the file name"
        )
    elif named is None and header_time is not None:
        notes.append("start time taken from the AudioMoth header")
    elif named is None:
        notes.append(NO_START_TIME)
    started_at = named if named is not None else header_time
    return RecordingTime(started_at, utc_offset, tuple(notes))


def start_time_from_name(stem: str) -> datetime | None:
    """The start time given by a YYYYMMDD_HHMMSS file name, or None."""
    return _file_name_time(stem)[0]


def audiomoth_comment(path: Path) -> str | None:
    """The INFO comment of a WAV file, where AudioMoth records its clock and time zone."""
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            return None
        while len(chunk_header := handle.read(8)) == 8:
            chunk_id = chunk_header[:4]
            (size,) = struct.unpack("<I", chunk_header[4:])
            padded = size + size % 2
            if chunk_id == b"LIST" and size <= _MAX_LIST_BYTES:
                body = handle.read(padded)[:size]
                comment = _info_comment(body) if body[:4] == b"INFO" else None
                if comment is not None:
                    return comment
            else:
                handle.seek(padded, 1)
    return None


def _info_comment(info: bytes) -> str | None:
    offset = 4
    while offset + 8 <= len(info):
        field_id = info[offset : offset + 4]
        (size,) = struct.unpack("<I", info[offset + 4 : offset + 8])
        if field_id == b"ICMT":
            return info[offset + 8 : offset + 8 + size].split(b"\0", 1)[0].decode("latin-1")
        offset += 8 + size + size % 2
    return None


def _file_name_time(stem: str) -> tuple[datetime | None, str | None]:
    match = _FILE_NAME_TIMESTAMP.search(stem)
    if match is None:
        return None, None
    try:
        return datetime.strptime(match.group(), "%Y%m%d_%H%M%S"), None
    except ValueError:
        return None, f"file name timestamp {match.group()} is not a valid date"


def _audiomoth_time(path: Path) -> tuple[datetime | None, str | None]:
    comment = audiomoth_comment(path)
    match = _AUDIOMOTH_TIME.search(comment) if comment is not None else None
    if match is None:
        return None, None
    try:
        started_at = datetime.strptime(match["clock"], "%H:%M:%S %d/%m/%Y")
    except ValueError:
        return None, None
    hours, minutes = int(match["hours"] or 0), int(match["minutes"] or 0)
    return started_at, f"{match['sign'] or '+'}{hours:02d}:{minutes:02d}"
