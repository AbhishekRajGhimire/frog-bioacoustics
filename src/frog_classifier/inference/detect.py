"""One detection run: resume, render in parallel, predict, store, and export."""

from __future__ import annotations

import multiprocessing
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch

from frog_classifier.training.fit import predict_probabilities, use_full_precision
from frog_classifier.training.saved import SavedRun, load_saved_run

from .export import ExportError, export_csv
from .recordings import file_head_sha256, find_recordings
from .rendering import RenderedRecording, RenderJob, render_for_detection, start_render_worker
from .store import DATABASE_FILE, DetectionStore, RecordingRow, ScanIdentity


MAX_DEFAULT_WORKERS = 8
# Rendered recordings wait in memory until predicted; this bounds how many per worker.
RECORDINGS_IN_FLIGHT_PER_WORKER = 2


class DetectionError(ValueError):
    pass


@dataclass(frozen=True)
class RecordingOutcome:
    path: str
    failed: bool
    windows: int
    detections: int
    message: str | None


@dataclass(frozen=True)
class DetectionSummary:
    """found, already_done, processed, audio_seconds, and the timings describe
    this run; failed, windows, and detections describe the whole record."""

    output_dir: Path
    threshold: float
    found: int
    already_done: int
    processed: int
    failed: int
    windows: int
    detections: int
    audio_seconds: float
    wall_seconds: float
    render_seconds: float
    predict_seconds: float
    csv_problem: str | None


@dataclass
class _RunTotals:
    processed: int = 0
    audio_seconds: float = 0.0
    render_seconds: float = 0.0
    predict_seconds: float = 0.0

    def add(self, row: RecordingRow) -> None:
        self.processed += 1
        if row.status == "ok" and row.duration_s is not None:
            self.audio_seconds += row.duration_s
        self.render_seconds += row.render_seconds
        self.predict_seconds += row.predict_seconds


def default_workers() -> int:
    """Half the logical CPUs, at most eight: rendering is CPU-bound, and the rest
    stay free for prediction and for whatever else the computer is doing."""
    return max(1, min(MAX_DEFAULT_WORKERS, (os.cpu_count() or 2) // 2))


def default_output_dir(results_root: Path, input_root: Path, run_name: str) -> Path:
    name = input_root.resolve().name
    if not name:
        raise DetectionError(
            f"{input_root} has no folder name to label its results with; "
            "name an output folder with --output, one for each card or deployment"
        )
    return results_root / f"{name}-{run_name}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def detect_calls(
    run_dir: Path,
    input_root: Path,
    output_dir: Path,
    *,
    device: torch.device,
    workers: int,
    on_start: Callable[[int, int], None] = lambda pending, done: None,
    on_recording: Callable[[RecordingOutcome], None] = lambda outcome: None,
    clock: Callable[[], datetime] = utc_now,
) -> DetectionSummary:
    """Detect calls in every recording beneath input_root not yet stored in output_dir.

    on_start receives the number of recordings to process and the number already
    done; on_recording receives each recording once it is stored.
    """
    if workers < 1:
        raise DetectionError("workers must be at least 1")
    input_root, output_dir = input_root.resolve(), output_dir.resolve()
    if not input_root.is_dir():
        raise DetectionError(f"input folder not found: {input_root}")
    if output_dir.is_relative_to(input_root):
        raise DetectionError(
            f"the output folder must not lie inside the input folder, which holds the source recordings: {output_dir}"
        )
    saved = load_saved_run(run_dir)
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    store = DetectionStore.open(output_dir / DATABASE_FILE, _scan_identity(saved, input_root), now=_timestamp(clock()))
    with store:
        found = find_recordings(input_root)
        done = store.finished_recordings()
        _refuse_changed_recordings(input_root, output_dir, found, done)
        jobs = [RenderJob(input_root, path, saved.preprocessing) for path in found if path not in done]
        on_start(len(jobs), len(found) - len(jobs))
        use_full_precision()
        model = saved.model.to(device)
        totals = _RunTotals()
        with closing(_render_in_order(jobs, workers)) as rendered_recordings:
            for rendered in rendered_recordings:
                prediction_started = time.perf_counter()
                probabilities = _predict(model, rendered.images, device)
                row = _recording_row(rendered, time.perf_counter() - prediction_started, _timestamp(clock()))
                store.save(row, rendered.window_starts, probabilities)
                totals.add(row)
                on_recording(RecordingOutcome(
                    path=row.path,
                    failed=row.status == "failed",
                    windows=row.windows,
                    detections=int(np.count_nonzero(probabilities >= saved.threshold)),
                    message=row.message,
                ))
        csv_problem = None
        try:
            export_csv(store, output_dir)
        except ExportError as error:
            csv_problem = str(error)
        recordings = store.recording_rows()
        detections = len(store.windows(minimum_probability=saved.threshold))
    return DetectionSummary(
        output_dir=output_dir,
        threshold=saved.threshold,
        found=len(found),
        already_done=len(found) - len(jobs),
        processed=totals.processed,
        failed=sum(row.status == "failed" for row in recordings),
        windows=sum(row.windows for row in recordings),
        detections=detections,
        audio_seconds=totals.audio_seconds,
        wall_seconds=time.perf_counter() - started,
        render_seconds=totals.render_seconds,
        predict_seconds=totals.predict_seconds,
        csv_problem=csv_problem,
    )


def _scan_identity(saved: SavedRun, input_root: Path) -> ScanIdentity:
    return ScanIdentity(
        input_root=input_root.as_posix(),
        model_run=saved.name,
        model_sha256=saved.model_sha256,
        threshold=saved.threshold,
        preprocessing_sha256=saved.preprocessing.sha256,
        chunk_seconds=saved.preprocessing.audio.chunk_seconds,
    )


def _refuse_changed_recordings(
    input_root: Path,
    output_dir: Path,
    found: list[str],
    done: dict[str, str | None],
) -> None:
    """Refuse to continue when a different file sits under a name already stored.

    Recorders on the same schedule write identical file names, so a second memory
    card read from the same drive would otherwise be skipped as already done.
    """
    changed = [path for path in found if path in done and file_head_sha256(input_root / path) != done[path]]
    if changed:
        raise DetectionError(
            f"{len(changed)} recordings in {input_root} differ from the recordings already stored "
            f"under the same names in {output_dir}, for example {changed[0]}; "
            "choose another output folder for these recordings"
        )


def _render_in_order(jobs: list[RenderJob], workers: int) -> Iterator[RenderedRecording]:
    """Rendered recordings in job order.

    With several workers, rendering runs ahead in other processes while this one
    predicts and stores.
    """
    workers = min(workers, len(jobs))
    if workers <= 1:
        yield from map(render_for_detection, jobs)
        return
    # Spawned rather than forked on every platform: forking a process that has
    # initialized CUDA is unsafe.
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=start_render_worker,
    )
    try:
        remaining = iter(jobs)
        in_flight = deque(
            executor.submit(render_for_detection, job)
            for job in islice(remaining, workers * RECORDINGS_IN_FLIGHT_PER_WORKER)
        )
        while in_flight:
            rendered = in_flight.popleft().result()
            for job in islice(remaining, 1):
                in_flight.append(executor.submit(render_for_detection, job))
            yield rendered
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _predict(model: torch.nn.Module, images: np.ndarray, device: torch.device) -> np.ndarray:
    if len(images) == 0:
        return np.empty(0)
    return predict_probabilities(model, images, device)


def _recording_row(rendered: RenderedRecording, predict_seconds: float, processed_at: str) -> RecordingRow:
    messages = ([rendered.error] if rendered.error is not None else []) + list(rendered.notes)
    return RecordingRow(
        path=rendered.path,
        recording_id=rendered.recording_id,
        started_at=None if rendered.started_at is None else rendered.started_at.isoformat(),
        utc_offset=rendered.utc_offset,
        size_bytes=rendered.size_bytes,
        audio_sha256=rendered.audio_sha256,
        head_sha256=rendered.head_sha256,
        duration_s=rendered.duration_s,
        windows=len(rendered.window_starts),
        status="failed" if rendered.error is not None else "ok",
        message="; ".join(messages) or None,
        render_seconds=rendered.render_seconds,
        predict_seconds=predict_seconds,
        processed_at=processed_at,
    )


def _timestamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
