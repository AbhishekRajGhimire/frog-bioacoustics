"""The detection record: one SQLite database per output folder.

Each recording is stored in a single transaction together with all of its
windows, so an interrupted run leaves only whole recordings behind and the
next run continues from them.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Iterator, Sequence


SCHEMA_VERSION = 1
DATABASE_FILE = "detections.sqlite"

_TABLES = (
    """CREATE TABLE scan (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        schema_version INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        input_root TEXT NOT NULL,
        model_run TEXT NOT NULL,
        model_sha256 TEXT NOT NULL,
        threshold REAL NOT NULL,
        preprocessing_sha256 TEXT NOT NULL,
        chunk_seconds INTEGER NOT NULL
    )""",
    """CREATE TABLE recordings (
        path TEXT PRIMARY KEY,
        recording_id TEXT NOT NULL,
        started_at TEXT,
        utc_offset TEXT,
        size_bytes INTEGER,
        audio_sha256 TEXT,
        head_sha256 TEXT,
        duration_s REAL,
        windows INTEGER NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('ok', 'failed')),
        message TEXT,
        render_seconds REAL NOT NULL,
        predict_seconds REAL NOT NULL,
        processed_at TEXT NOT NULL
    )""",
    """CREATE TABLE windows (
        path TEXT NOT NULL REFERENCES recordings (path),
        start_s INTEGER NOT NULL,
        probability REAL NOT NULL,
        PRIMARY KEY (path, start_s)
    )""",
)


class StoreError(ValueError):
    pass


@dataclass(frozen=True)
class ScanIdentity:
    """What a database's results depend on; one database never mixes two of these."""

    input_root: str
    model_run: str
    model_sha256: str
    threshold: float
    preprocessing_sha256: str
    chunk_seconds: int


@dataclass(frozen=True)
class RecordingRow:
    path: str
    recording_id: str
    started_at: str | None
    utc_offset: str | None
    size_bytes: int | None
    audio_sha256: str | None
    head_sha256: str | None
    duration_s: float | None
    windows: int
    status: str
    message: str | None
    render_seconds: float
    predict_seconds: float
    processed_at: str


_IDENTITY_COLUMNS = tuple(field.name for field in fields(ScanIdentity))
_RECORDING_COLUMNS = tuple(field.name for field in fields(RecordingRow))


class DetectionStore:
    def __init__(self, connection: sqlite3.Connection, identity: ScanIdentity, created_at: str) -> None:
        self._connection = connection
        self.identity = identity
        self.created_at = created_at

    @classmethod
    def open(cls, path: Path, identity: ScanIdentity, *, now: str) -> DetectionStore:
        """Create the database for a new scan, or reopen one made for exactly this scan."""
        # Transactions are explicit: autocommit mode, with BEGIN and COMMIT issued here.
        connection = sqlite3.connect(path, isolation_level=None)
        try:
            tables = _table_names(connection, path)
            connection.execute("PRAGMA foreign_keys = ON")
            if not tables:
                created_at = _create(connection, identity, now)
            elif "scan" not in tables:
                raise StoreError(f"{path} is not a detection database")
            else:
                stored, created_at = _read_scan(connection, path)
                changed = [name for name in _IDENTITY_COLUMNS if getattr(stored, name) != getattr(identity, name)]
                if changed:
                    raise StoreError(f"{path.parent} holds results for a different {', '.join(changed)}")
        except StoreError as error:
            connection.close()
            raise StoreError(f"{error}; choose another output folder") from error
        except BaseException:
            connection.close()
            raise
        return cls(connection, identity, created_at)

    @classmethod
    def open_read_only(cls, path: Path) -> DetectionStore:
        """Open an existing detection database for reading only, taking its identity from the database."""
        if not path.is_file():
            raise StoreError(f"{path} does not exist; run the detection command first")
        # A non-empty rollback journal means a transaction is open or was cut off;
        # a reader must not see the database until the next writer has settled it.
        journal = path.with_name(f"{path.name}-journal")
        if journal.is_file() and journal.stat().st_size > 0:
            raise StoreError(
                f"a detection run is still writing {path} or was interrupted; "
                "let it finish, or run the same detection command again, before reading it"
            )
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            if "scan" not in _table_names(connection, path):
                raise StoreError(f"{path} is not a detection database")
            identity, created_at = _read_scan(connection, path)
        except BaseException:
            connection.close()
            raise
        return cls(connection, identity, created_at)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> DetectionStore:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def finished_recordings(self) -> dict[str, str | None]:
        """The head checksum of every recording stored successfully, by path; a
        failed recording is not finished and is retried."""
        rows = self._connection.execute("SELECT path, head_sha256 FROM recordings WHERE status = 'ok'")
        return dict(rows.fetchall())

    def save(self, recording: RecordingRow, window_starts: Sequence[int], probabilities: Sequence[float]) -> None:
        """Store one recording with all of its windows, replacing any earlier attempt."""
        if not recording.windows == len(window_starts) == len(probabilities):
            raise ValueError("a recording's window count must match its window starts and probabilities")
        columns = ", ".join(_RECORDING_COLUMNS)
        placeholders = ", ".join("?" * len(_RECORDING_COLUMNS))
        with _transaction(self._connection):
            self._connection.execute("DELETE FROM windows WHERE path = ?", (recording.path,))
            self._connection.execute("DELETE FROM recordings WHERE path = ?", (recording.path,))
            self._connection.execute(
                f"INSERT INTO recordings ({columns}) VALUES ({placeholders})",
                astuple(recording),
            )
            self._connection.executemany(
                "INSERT INTO windows (path, start_s, probability) VALUES (?, ?, ?)",
                [
                    (recording.path, int(start), float(probability))
                    for start, probability in zip(window_starts, probabilities, strict=True)
                ],
            )

    def recording_rows(self) -> list[RecordingRow]:
        query = f"SELECT {', '.join(_RECORDING_COLUMNS)} FROM recordings ORDER BY path"
        return [RecordingRow(*row) for row in self._connection.execute(query)]

    def windows(self, *, minimum_probability: float = 0.0) -> list[tuple[str, int, float]]:
        """(path, start_s, probability) of every window at or above minimum_probability."""
        return list(self._connection.execute(
            "SELECT path, start_s, probability FROM windows WHERE probability >= ? ORDER BY path, start_s",
            (minimum_probability,),
        ))


@contextmanager
def _transaction(connection: sqlite3.Connection) -> Iterator[None]:
    connection.execute("BEGIN")
    try:
        yield
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")


def _table_names(connection: sqlite3.Connection, path: Path) -> set[str]:
    try:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    except sqlite3.DatabaseError as error:
        raise StoreError(f"{path} is not a detection database ({error})") from error
    return {name for (name,) in rows}


def _create(connection: sqlite3.Connection, identity: ScanIdentity, now: str) -> str:
    columns = ", ".join(_IDENTITY_COLUMNS)
    placeholders = ", ".join("?" * len(_IDENTITY_COLUMNS))
    with _transaction(connection):
        for statement in _TABLES:
            connection.execute(statement)
        connection.execute(
            f"INSERT INTO scan (id, schema_version, created_at, {columns}) VALUES (1, ?, ?, {placeholders})",
            (SCHEMA_VERSION, now, *astuple(identity)),
        )
    return now


def _read_scan(connection: sqlite3.Connection, path: Path) -> tuple[ScanIdentity, str]:
    """The stored scan identity and creation time."""
    (version,) = connection.execute("SELECT schema_version FROM scan").fetchone()
    if version != SCHEMA_VERSION:
        raise StoreError(f"{path} uses detection schema {version}, but this code reads schema {SCHEMA_VERSION}")
    row = connection.execute(f"SELECT created_at, {', '.join(_IDENTITY_COLUMNS)} FROM scan").fetchone()
    return ScanIdentity(*row[1:]), row[0]
