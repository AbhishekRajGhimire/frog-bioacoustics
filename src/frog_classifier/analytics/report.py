"""The analysis report in Markdown, and the summary lines the command prints."""

from __future__ import annotations

from .activity import Activity, ActivityBin
from .figures import BY_DATE_FIGURE, CLOCK_FIGURE, TIME_OF_DAY_FIGURE


FIGURES = (
    (TIME_OF_DAY_FIGURE, "Calling share by hour of day"),
    (CLOCK_FIGURE, "Calling share on a 24-hour clock"),
    (BY_DATE_FIGURE, "Calling share by date"),
)
_TABLE_HEADER = (
    "| {key} | Recorded hours | Recordings | Detected windows | Calls in recorded time |",
    "| --- | ---: | ---: | ---: | ---: |",
)


def summary_lines(activity: Activity) -> list[str]:
    """The headline facts, shared by the report and the command's output."""
    recorded_dates = [day for day, bin_ in activity.by_date.items() if bin_.recorded_windows]
    if not recorded_dates:
        return ["No recording could be placed in time, so there is no activity to report."]
    total = activity.total
    lines = [
        f"Recorded {total.recorded_minutes / 60:.1f} hours in {activity.coverage.recordings} recordings "
        f"on {len(recorded_dates)} of the {len(activity.by_date)} dates from {recorded_dates[0]} to {recorded_dates[-1]}.",
    ]
    hours = f"Hours of day recorded: {_hour_ranges(activity, recorded=True)}."
    unrecorded = _hour_ranges(activity, recorded=False)
    if unrecorded:
        hours += f" Not recorded: {unrecorded}, so activity then is unknown."
    lines.append(hours)
    lines.append(
        f"Calls were detected in {total.calling_share:.1%} of recorded time "
        f"({total.detected_windows:,} of {total.recorded_windows:,} windows of {activity.identity.chunk_seconds} seconds) "
        f"at threshold {activity.threshold:.3f}."
    )
    # max keeps the earliest hour when shares tie.
    peak = max(range(24), key=lambda hour: _share_or_minus_one(activity.by_hour[hour]))
    lines.append(
        f"Highest calling share: {peak:02d}:00 to {peak:02d}:59, "
        f"{activity.by_hour[peak].calling_share:.1%} of recorded time."
    )
    if activity.time_basis.offset is None:
        lines.append("Times are the recorders' own clock, because no recording states its UTC offset.")
    else:
        lines.append(
            f"Times are in {activity.time_basis.label} for every recording, "
            "so daylight-saving changes cannot shift them."
        )
    return lines


def render_report(activity: Activity, record_name: str) -> str:
    identity, coverage = activity.identity, activity.coverage
    lines = [f"# Calling activity: {record_name}", "", "## Summary", ""]
    lines += [f"- {line}" for line in summary_lines(activity)]
    lines += ["", "## Figures", ""]
    for file_name, title in FIGURES:
        lines += [f"![{title}]({file_name})", ""]
    lines += ["## By hour of day", "", _TABLE_HEADER[0].format(key="Hour"), _TABLE_HEADER[1]]
    lines += [_bin_row(f"{hour:02d}:00", bin_) for hour, bin_ in enumerate(activity.by_hour)]
    lines += ["", "## By date", "", _TABLE_HEADER[0].format(key="Date"), _TABLE_HEADER[1]]
    lines += [_bin_row(day.isoformat(), bin_) for day, bin_ in activity.by_date.items()]
    lines += [
        "",
        "## Threshold",
        "",
        f"A window counts as a detection when its probability is at or above {activity.threshold:.3f}; "
        f"the model's own threshold is {identity.threshold:.3f}.",
        "The same totals at other thresholds show how much the result depends on that choice.",
        "",
        "| Threshold | Detected windows | Calls in recorded time |",
        "| ---: | ---: | ---: |",
    ]
    lines += [
        f"| {row.threshold:.3f} | {row.detected_windows} | {_share(row.calling_share)} |"
        for row in activity.thresholds
    ]
    lines += [
        "",
        "## Provenance",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Detection record | {record_name} |",
        f"| Record created | {activity.record_created_at} |",
        f"| Input folder | {identity.input_root} |",
        f"| Model run | {identity.model_run} |",
        f"| Model SHA-256 | `{identity.model_sha256}` |",
        f"| Preprocessing SHA-256 | `{identity.preprocessing_sha256}` |",
        f"| Model threshold | {identity.threshold:.8f} |",
        f"| Analysis threshold | {activity.threshold:.8f} |",
        f"| Time basis | {activity.time_basis.label} |",
        f"| UTC offsets the recorders state | {', '.join(coverage.stated_offsets) or 'none'} |",
        f"| Recordings analyzed | {coverage.recordings} |",
        f"| Failed recordings | {coverage.failed_recordings} |",
        f"| Recordings without a start time, left out | {coverage.recordings_without_time} |",
        f"| Recordings without a UTC offset, taken to use the time basis | {coverage.recordings_without_offset} |",
        f"| Windows that could not be placed in time | {coverage.unplaced_windows} |",
        "",
    ]
    return "\n".join(lines)


def _bin_row(label: str, bin_: ActivityBin) -> str:
    return (
        f"| {label} | {bin_.recorded_minutes / 60:.2f} | {bin_.recordings} | "
        f"{bin_.detected_windows} | {_share(bin_.calling_share)} |"
    )


def _share(share: float | None) -> str:
    return "not recorded" if share is None else f"{share:.1%}"


def _share_or_minus_one(bin_: ActivityBin) -> float:
    return -1.0 if bin_.calling_share is None else bin_.calling_share


def _hour_ranges(activity: Activity, *, recorded: bool) -> str:
    """Runs of consecutive hours that were, or were not, recorded, such as 00:00 to 11:59."""
    runs: list[list[int]] = []
    for hour, bin_ in enumerate(activity.by_hour):
        if (bin_.recorded_windows > 0) != recorded:
            continue
        if runs and runs[-1][1] == hour - 1:
            runs[-1][1] = hour
        else:
            runs.append([hour, hour])
    return ", ".join(f"{start:02d}:00 to {end:02d}:59" for start, end in runs)
