"""Replace generated files without ever leaving half a file behind."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


class FileInUseError(Exception):
    pass


def replace_file(path: Path, write: Callable[[Path], None]) -> None:
    """Call write with a temporary path beside path, then swap the result into place.

    A reader sees either the old file or the new one. If path is open in another
    program and cannot be replaced, the temporary file is removed and the old
    file stays as it was.
    """
    staging = path.with_name(f".{path.name}.partial")
    write(staging)
    try:
        os.replace(staging, path)
    except PermissionError as error:
        staging.unlink()
        raise FileInUseError(
            f"{path.name} could not be replaced, probably because it is open in another program; "
            "close it and run the same command again"
        ) from error
