"""Save uploaded recordings into a deployment's upload folder without overwriting any."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePath, PureWindowsPath
from typing import Sequence

from frog_classifier.files import replace_file
from frog_classifier.inference.recordings import AUDIO_SUFFIXES


class UploadError(ValueError):
    pass


@dataclass(frozen=True)
class SavedUploads:
    saved: tuple[str, ...]
    skipped: tuple[str, ...]


def save_uploads(files: Sequence[tuple[str, bytes]], folder: Path) -> SavedUploads:
    """Save each (file name, content) in folder.

    A file already there with the same content is skipped. One with different
    content is refused, and then nothing is saved, so a recording is never
    overwritten.
    """
    # Browsers may send folder parts, with either separator; only the file name is kept.
    named = [(PureWindowsPath(name).name, content) for name, content in files]
    unsupported = [name for name, _ in named if PurePath(name).suffix.lower() not in AUDIO_SUFFIXES]
    if unsupported:
        raise UploadError(f"only .wav and .mp3 recordings can be scanned, not {', '.join(unsupported)}")
    repeated = [name for name, count in Counter(name.lower() for name, _ in named).items() if count > 1]
    if repeated:
        raise UploadError(f"{', '.join(repeated)} is in this upload more than once")
    conflicts = [
        name for name, content in named
        if (folder / name).is_file() and (folder / name).read_bytes() != content
    ]
    if conflicts:
        raise UploadError(
            f"{folder} already holds different recordings named {', '.join(conflicts)}; "
            "rename them or use another name for these recordings"
        )
    folder.mkdir(parents=True, exist_ok=True)
    saved, skipped = [], []
    for name, content in named:
        target = folder / name
        if target.is_file():
            skipped.append(name)
            continue
        replace_file(target, lambda staging, content=content: staging.write_bytes(content))
        saved.append(name)
    return SavedUploads(tuple(saved), tuple(skipped))
