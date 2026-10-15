"""Frog call scanner: find Litoria aurea calls in field recordings and see when frogs call.

Start it with: uv run streamlit run classifier_app/app.py
Scans run in their own process, so this page never loads the model itself.
"""

from __future__ import annotations

import io
from datetime import date, datetime, timezone
from pathlib import Path

import streamlit as st

from frog_classifier.analytics.activity import Activity, ActivityBin, ActivityError, PlacedWindow, read_activity
from frog_classifier.analytics.figures import write_by_date_figure, write_clock_figure, write_time_of_day_figure
from frog_classifier.analytics.report import coverage_warnings, headline_lines
from frog_classifier.analytics.timebase import OffsetError
from frog_classifier.app import dialog
from frog_classifier.app.deployments import (
    NAME_LIMIT,
    Deployment,
    DeploymentError,
    RecordingsPreview,
    clean_name,
    list_deployments,
    live_scan,
    preview_recordings,
    start_scan,
)
from frog_classifier.app.export import results_zip
from frog_classifier.app.listening import ListeningError, WindowView, load_window, model_preprocessing
from frog_classifier.app.models import ModelError, ModelInfo, newest_model
from frog_classifier.app.paths import AppPaths, app_paths
from frog_classifier.app.scans import ScanStatus, request_stop
from frog_classifier.app.uploads import UploadError, save_uploads
from frog_classifier.inference.store import DATABASE_FILE, DetectionStore, StoreError


REPO_ROOT = Path(__file__).resolve().parents[1]
VIEW_SCAN = "Scan recordings"
VIEW_RESULTS = "Results"
VIEW_ABOUT = "About"
SOURCE_FOLDER = "A folder or memory card"
SOURCE_UPLOAD = "Upload a few files"
LOOK_HOUR = "An hour of day"
LOOK_DATE = "A date"
FIGURE_TITLES = ("By hour of day", "24-hour clock", "By date")
FIGURE_WRITERS = (write_time_of_day_figure, write_clock_figure, write_by_date_figure)
LISTEN_LIMIT = 100
DEVICES = {"cuda": "Using the graphics card.", "cpu": "Using the processor; a graphics card would be faster."}
VALIDATION_NOTICE = (
    "This detector was trained and checked on recordings from one site, IWS076, made between midnight and noon "
    "from January to April 2022. At other sites, seasons, or hours, listen to a sample of detections before "
    "relying on the numbers."
)
ABOUT = """
This app finds the calls of the green and golden bell frog, *Litoria aurea*, in field recordings.

**Scan recordings** cuts every recording into five-second windows and asks the detector whether each one holds a call.
A whole memory card takes a few minutes.
A scan can be stopped and continued later, and closing the app stops it.

**Results** shows when frogs called:

- *Calls in recorded time* is the share of the recorded five-second windows that hold a call.
- Hours and dates that were never recorded are hatched grey and marked not recorded: nothing is known about them, so they are not zero.
- Every time uses one fixed UTC offset, so daylight-saving changes cannot shift the pattern.
- *Look closer* lists the windows behind any hour or date and plays them, so every number can be checked by ear.
- *Download results* saves the report, charts, and tables as one zip file.
"""


def main() -> None:
    st.set_page_config(page_title="Frog call scanner", page_icon=":frog:", layout="wide")
    paths = app_paths(REPO_ROOT)
    view = st.sidebar.radio("View", (VIEW_SCAN, VIEW_RESULTS, VIEW_ABOUT), key="view")
    if view == VIEW_SCAN:
        scan_view(paths)
    elif view == VIEW_RESULTS:
        results_view(paths)
    else:
        about_view(paths)


# Scanning


def scan_view(paths: AppPaths) -> None:
    st.title("Scan recordings")
    try:
        model = newest_model(paths.models)
    except ModelError as error:
        st.error(f"Scanning is not possible yet: {error}.")
        return
    problem = st.session_state.pop("scan_problem", None)
    if problem is not None:
        st.error(f"Cannot scan: {problem}.")
    deployments = list_deployments(paths.detections, now=_now())
    live = live_scan(deployments)
    if live is not None:
        st.session_state["scan_record"] = live.record_dir.name
    chosen = next((d for d in deployments if d.record_dir.name == st.session_state.get("scan_record")), None)
    if chosen is None:
        _scan_form(paths, model)
    elif chosen.scanning:
        _live_progress(chosen.record_dir.name, chosen.name, paths)
    else:
        _scan_outcome(paths, model, chosen)


def _scan_form(paths: AppPaths, model: ModelInfo) -> None:
    st.write("Choose the recordings from one recorder deployment or memory card, give them a name, and press Scan.")
    source = st.radio("Where are the recordings?", (SOURCE_FOLDER, SOURCE_UPLOAD), horizontal=True, key="source")
    if source == SOURCE_FOLDER:
        ready = _folder_choice()
    else:
        uploads = st.file_uploader(
            "Recordings",
            type=["wav", "mp3"],
            accept_multiple_files=True,
            key="uploads",
            help="Uploading copies each file into the app. For a whole memory card, choose its folder instead.",
        )
        ready = bool(uploads)
    st.text_input(
        "Name for these recordings",
        key="deployment_name",
        max_chars=NAME_LIMIT,
        help="For example the site and card number, such as IWS076 card 3. Results are kept under this name.",
    )
    st.button("Scan", key="scan", type="primary", disabled=not ready, on_click=_start_from_form, args=(paths, model))
    st.caption(f"Detector {model.name}, threshold {model.threshold:.3f}.")


def _folder_choice() -> bool:
    folder_column, browse_column = st.columns([5, 1], vertical_alignment="bottom")
    folder_column.text_input("Folder", key="folder", placeholder=r"For example E:\ for a memory card, or D:\recordings\site-a")
    browse_column.button("Browse...", key="browse", on_click=_browse)
    text = _folder_text()
    if not text:
        st.info("Paste the folder's location, or press Browse to pick it. A memory card usually appears as a drive such as `E:\\`.")
        return False
    try:
        preview = preview_recordings(Path(text))
    except DeploymentError as error:
        st.error(f"{str(error).capitalize()}.")
        return False
    if preview.count == 0:
        st.warning("There are no .wav or .mp3 recordings in this folder or its subfolders.")
        return False
    st.success(_preview_text(preview))
    if st.session_state.get("named_folder") != text:
        # A new folder suggests its own name; a name typed for this folder is kept.
        st.session_state["named_folder"] = text
        st.session_state["deployment_name"] = Path(text).resolve().name
    return True


def _preview_text(preview: RecordingsPreview) -> str:
    text = f"Found {preview.count:,} recording{'s' if preview.count != 1 else ''}"
    if preview.first_day is not None:
        text += f", dated {_day(preview.first_day)}"
        if preview.last_day != preview.first_day:
            text += f" to {_day(preview.last_day)}"
    text += "."
    if preview.undated:
        text += f" {preview.undated:,} {'has' if preview.undated == 1 else 'have'} no date in the file name."
    return text


@st.fragment(run_every=1.0)
def _live_progress(folder: str, name: str, paths: AppPaths) -> None:
    deployment = next((d for d in list_deployments(paths.detections, now=_now()) if d.record_dir.name == folder), None)
    if deployment is None or not deployment.scanning:
        st.rerun()
    status = deployment.status
    st.subheader(f"Scanning {name}")
    if status.state == "starting":
        st.progress(0.0, text="Starting: loading the detector...")
    else:
        fraction = status.done / status.found if status.found else 1.0
        st.progress(fraction, text=f"Scanned {status.done:,} of {status.found:,} recordings{_time_left(status)}")
    if status.failed:
        st.warning(f"{status.failed:,} recordings could not be read so far.")
    if status.device is not None:
        st.caption(DEVICES.get(status.device, status.device))
    st.button(
        "Stop scan",
        key="stop",
        on_click=request_stop,
        args=(deployment.record_dir,),
        help="Finished recordings are kept, and the scan can be continued later.",
    )


def _scan_outcome(paths: AppPaths, model: ModelInfo, deployment: Deployment) -> None:
    status = deployment.status
    st.subheader(deployment.name)
    if status is None or status.state == "finished":
        counted = f": {status.found:,} recordings" if status is not None else ""
        st.success(f"The scan of {deployment.name} finished{counted}.")
        if status is not None and status.failed:
            st.warning(f"{status.failed:,} recordings could not be read. Continue scan tries them again.")
        if status is not None and status.message:
            st.warning(status.message)
    elif status.state in ("stopped", "interrupted"):
        how = "was stopped" if status.state == "stopped" else "ended unexpectedly, for example because the app was closed,"
        progress = f"after {status.done:,} of {status.found:,} recordings" if status.found else "before it began scanning"
        st.warning(
            f"The scan of {deployment.name} {how} {progress}. "
            "Finished recordings are kept, and continuing picks up where it stopped."
        )
    else:
        st.error(f"The scan of {deployment.name} could not run: {status.message}")
    columns = st.columns(3)
    if deployment.results_ready:
        columns[0].button(
            "View results", key="view_results", type="primary", on_click=_open_results, args=(deployment.record_dir.name,),
        )
    if status is not None and (status.state != "finished" or status.failed):
        columns[1].button(
            "Continue scan",
            key="continue",
            on_click=_start,
            args=(paths, model, Path(deployment.input_root), deployment.name),
        )
    columns[2].button("Scan other recordings", key="new_scan", on_click=_forget_scan)


def _time_left(status: ScanStatus) -> str:
    if status.processed < 3:
        return ""
    elapsed = (_now() - datetime.fromisoformat(status.started_at)).total_seconds()
    seconds = elapsed / status.processed * (status.found - status.done)
    if seconds < 60:
        return ", less than a minute left"
    minutes = round(seconds / 60)
    return f", about {minutes} minute{'s' if minutes != 1 else ''} left"


def _start_from_form(paths: AppPaths, model: ModelInfo) -> None:
    name = st.session_state.get("deployment_name", "")
    try:
        if st.session_state.get("source") == SOURCE_UPLOAD:
            folder = paths.uploads / clean_name(name)
            uploads = st.session_state.get("uploads") or []
            save_uploads([(upload.name, upload.getvalue()) for upload in uploads], folder)
        else:
            folder = Path(_folder_text())
    except (DeploymentError, UploadError) as error:
        st.session_state["scan_problem"] = str(error)
        return
    _start(paths, model, folder, name)


def _start(paths: AppPaths, model: ModelInfo, folder: Path, name: str) -> None:
    try:
        record_dir = start_scan(paths, model, folder, name, now=_now())
    except DeploymentError as error:
        st.session_state["scan_problem"] = str(error)
        return
    st.session_state["scan_record"] = record_dir.name


def _browse() -> None:
    chosen = dialog.choose_folder()
    if chosen:
        st.session_state["folder"] = str(Path(chosen))


def _folder_text() -> str:
    return st.session_state.get("folder", "").strip().strip('"')


def _forget_scan() -> None:
    st.session_state.pop("scan_record", None)


def _open_results(folder: str) -> None:
    st.session_state["view"] = VIEW_RESULTS
    st.session_state["deployment"] = folder


def _open_scan(folder: str) -> None:
    st.session_state["view"] = VIEW_SCAN
    st.session_state["scan_record"] = folder


# Results


def results_view(paths: AppPaths) -> None:
    st.title("Results")
    deployments = list_deployments(paths.detections, now=_now())
    if not deployments:
        st.info("Nothing has been scanned yet. Start with Scan recordings.")
        return
    by_folder = {deployment.record_dir.name: deployment for deployment in deployments}
    if st.session_state.get("deployment") not in by_folder:
        st.session_state["deployment"] = deployments[0].record_dir.name
    names = [deployment.name for deployment in deployments]
    folder = st.selectbox(
        "Recordings",
        list(by_folder),
        key="deployment",
        format_func=lambda folder: _deployment_label(by_folder[folder], names),
    )
    deployment = by_folder[folder]
    if deployment.scanning:
        status = deployment.status
        st.info(
            f"{deployment.name} is still being scanned ({status.done:,} of {status.found:,} recordings so far). "
            "Its results appear here when the scan finishes."
        )
        return
    if not deployment.results_ready:
        st.warning(
            f"The scan of {deployment.name} did not finish, so its results would be incomplete. "
            "Continue it under Scan recordings."
        )
        st.button("Go to Scan recordings", key="go_scan", on_click=_open_scan, args=(folder,))
        return
    _results(paths, deployment)


def _results(paths: AppPaths, deployment: Deployment) -> None:
    folder = deployment.record_dir.name
    database = deployment.record_dir / DATABASE_FILE
    try:
        with DetectionStore.open_read_only(database) as store:
            identity = store.identity
        stamp = database.stat().st_mtime_ns
    except (OSError, StoreError) as error:
        st.error(f"The results of {deployment.name} cannot be read: {error}.")
        return
    with st.expander("Threshold and time zone"):
        threshold = st.slider(
            "Detection threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(identity.threshold),
            step=0.001,
            format="%.3f",
            key=f"threshold:{folder}",
            help="A window counts as a call when the detector's probability is at or above this. "
            "The default is the detector's own threshold.",
        )
        offset = st.text_input(
            "UTC offset",
            key=f"offset:{folder}",
            placeholder="The recorders' own, such as +11:00",
            help="One fixed UTC offset for every time, such as +10:00 for Australian Eastern Standard Time.",
        ).strip() or None
    try:
        activity, figures = _analysis(str(deployment.record_dir), stamp, threshold, offset)
    except (StoreError, OffsetError, ActivityError) as error:
        st.error(f"The results of {deployment.name} cannot be shown: {error}.")
        return
    st.subheader(deployment.name)
    st.markdown("\n".join(f"- {line}" for line in headline_lines(activity)))
    for warning in coverage_warnings(activity):
        st.warning(warning)
    st.caption(
        f"Detector {identity.model_run} at threshold {activity.threshold:.3f}; its own threshold is "
        f"{identity.threshold:.3f}. Times are in {activity.time_basis.label}. Recordings from {identity.input_root}."
    )
    st.info(VALIDATION_NOTICE)
    for tab, figure in zip(st.tabs(FIGURE_TITLES), figures):
        tab.image(figure)
    st.markdown("#### How much the threshold matters")
    st.dataframe(
        [
            {
                "Threshold": f"{row.threshold:.3f}",
                "Windows with calls": f"{row.detected_windows:,}",
                "Calls in recorded time": "not recorded" if row.calling_share is None else f"{row.calling_share:.1%}",
            }
            for row in activity.thresholds
        ],
        hide_index=True,
    )
    _look_closer(paths, deployment, activity)
    st.download_button(
        "Download results (zip)",
        key="download",
        type="primary",
        file_name=f"{folder}.zip",
        mime="application/zip",
        data=lambda: results_zip(deployment.record_dir, threshold=threshold, utc_offset=offset),
    )


def _look_closer(paths: AppPaths, deployment: Deployment, activity: Activity) -> None:
    folder = deployment.record_dir.name
    st.markdown("#### Look closer")
    if not activity.windows:
        st.info("No recording has a usable date and time, so there are no hours or dates to look at.")
        return
    st.caption("Pick an hour of day or a date to see every five-second window behind its bar, and listen to any of them.")
    mode = st.radio("Look at", (LOOK_HOUR, LOOK_DATE), horizontal=True, key=f"look:{folder}")
    if mode == LOOK_HOUR:
        hours = [hour for hour, bin_ in enumerate(activity.by_hour) if bin_.recorded_windows]
        hour = st.selectbox(
            "Hour of day",
            hours,
            key=f"hour:{folder}",
            format_func=lambda hour: f"{hour:02d}:00 to {hour:02d}:59, {_bin_text(activity.by_hour[hour])}",
        )
        windows, chosen = [window for window in activity.windows if window.hour == hour], f"{hour:02d}"
    else:
        days = [day for day, bin_ in activity.by_date.items() if bin_.recorded_windows]
        day = st.selectbox(
            "Date",
            days,
            key=f"day:{folder}",
            format_func=lambda day: f"{_day(day)}, {_bin_text(activity.by_date[day])}",
        )
        windows, chosen = [window for window in activity.windows if window.date == day], day.isoformat()
    windows.sort(key=lambda window: (not activity.is_detected(window), -window.probability, window.start))
    st.dataframe(
        [
            {
                "Date": f"{window.start:%Y-%m-%d}",
                "Time": f"{window.start:%H:%M:%S}",
                "Call": "yes" if activity.is_detected(window) else "",
                "Probability": f"{window.probability:.3f}",
                "Recording": window.recording_path,
                "Seconds in": window.window_start_s,
            }
            for window in windows
        ],
        hide_index=True,
        height=240,
    )
    index = st.selectbox(
        "Listen to",
        list(range(min(len(windows), LISTEN_LIMIT))),
        key=f"listen:{folder}:{chosen}",
        format_func=lambda index: _window_label(windows[index], activity),
        help=f"The first {LISTEN_LIMIT} windows in the table, calls first.",
    )
    window = windows[index]
    try:
        view = _window_view(
            deployment.input_root, window.recording_path, window.window_start_s, str(paths.models), deployment.model_run,
        )
    except ListeningError as error:
        st.warning(f"{str(error).capitalize()}.")
        return
    sound_column, image_column = st.columns([1, 2])
    sound_column.audio(view.wav, format="audio/wav")
    sound_column.caption("Played at full loudness, so faint calls can be heard.")
    image_column.image(
        view.image,
        caption="The spectrogram the detector saw: low pitches at the bottom, five seconds from left to right.",
        width="stretch",
    )


@st.cache_data(max_entries=16, show_spinner="Reading the results...")
def _analysis(record_dir: str, stamp: int, threshold: float, utc_offset: str | None) -> tuple[Activity, list[bytes]]:
    """The activity and its three figures; stamp, the database's modification time,
    makes a changed record a new cache entry."""
    with DetectionStore.open_read_only(Path(record_dir) / DATABASE_FILE) as store:
        activity = read_activity(store, threshold=threshold, utc_offset=utc_offset)
    figures = []
    for write in FIGURE_WRITERS:
        buffer = io.BytesIO()
        write(buffer, activity)
        figures.append(buffer.getvalue())
    return activity, figures


@st.cache_data(max_entries=32, show_spinner="Loading the recording...")
def _window_view(input_root: str, recording_path: str, start_s: int, models_root: str, model_run: str) -> WindowView:
    config = model_preprocessing(Path(models_root), model_run)
    return load_window(Path(input_root) / recording_path, start_s, config)


def _deployment_label(deployment: Deployment, names: list[str]) -> str:
    label = deployment.name
    if names.count(deployment.name) > 1:
        label += f" ({deployment.model_run})"
    if deployment.scanning:
        label += " (scanning)"
    elif not deployment.results_ready:
        label += " (not finished)"
    return label


def _day(day: date) -> str:
    """A date as people write it, such as 3 April 2022."""
    return f"{day.day} {day:%B %Y}"


def _bin_text(bin_: ActivityBin) -> str:
    return f"calls in {bin_.calling_share:.1%} of {bin_.recorded_windows:,} windows"


def _window_label(window: PlacedWindow, activity: Activity) -> str:
    call = ", call" if activity.is_detected(window) else ""
    return f"{window.start:%d %b %Y %H:%M:%S}, probability {window.probability:.3f}{call}"


# About


def about_view(paths: AppPaths) -> None:
    st.title("About")
    st.markdown(ABOUT)
    try:
        model = newest_model(paths.models)
    except ModelError as error:
        st.warning(f"{str(error).capitalize()}.")
    else:
        st.write(f"Detector in use: {model.name}, threshold {model.threshold:.3f}.")
    st.info(VALIDATION_NOTICE)


def _now() -> datetime:
    return datetime.now(timezone.utc)


main()
