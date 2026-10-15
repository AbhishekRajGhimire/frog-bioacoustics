"""The activity figures: calling share by hour of day, on a 24-hour clock, and by date.

Hours and dates without recordings are drawn as hatched grey "not recorded"
areas, so missing effort can never look like an absence of calls.
"""

from __future__ import annotations

import math
from datetime import datetime, time, timedelta
from pathlib import Path

from matplotlib import dates as mdates
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from .activity import Activity


TIME_OF_DAY_FIGURE = "time_of_day.png"
CLOCK_FIGURE = "clock.png"
BY_DATE_FIGURE = "by_date.png"

_CALLS = "tab:green"
_EFFORT = "tab:gray"
_UNRECORDED = {"facecolor": "0.93", "edgecolor": "0.75", "hatch": "//", "linewidth": 0}
_DPI = 120


def write_time_of_day_figure(path: Path, activity: Activity) -> None:
    figure = Figure(figsize=(9, 5.5), layout="constrained")
    share_axes, effort_axes = figure.subplots(2, 1, sharex=True, height_ratios=(3, 1))
    recorded = [hour for hour, bin_ in enumerate(activity.by_hour) if bin_.recorded_windows]
    for hour in range(24):
        if hour not in recorded:
            share_axes.axvspan(hour - 0.5, hour + 0.5, **_UNRECORDED)
    shares = [100 * activity.by_hour[hour].calling_share for hour in recorded]
    share_axes.bar(recorded, shares, width=0.8, color=_CALLS)
    share_axes.set(ylim=(0, _share_limit(shares)), ylabel="Calls in recorded time")
    share_axes.yaxis.set_major_formatter(PercentFormatter(100))
    share_axes.legend(handles=_legend(), loc="upper right")
    share_axes.grid(axis="y", alpha=0.3)
    effort_axes.bar(range(24), [bin_.recorded_minutes / 60 for bin_ in activity.by_hour], width=0.8, color=_EFFORT)
    effort_axes.set(ylabel="Recorded hours", xlabel=f"Hour of day ({activity.time_basis.label})", xlim=(-0.5, 23.5))
    effort_axes.set_xticks(range(24), labels=[f"{hour:02d}" for hour in range(24)])
    figure.suptitle(f"Calling activity by hour of day, threshold {activity.threshold:.3f}")
    figure.savefig(path, dpi=_DPI, format="png")


def write_clock_figure(path: Path, activity: Activity) -> None:
    figure = Figure(figsize=(6.5, 6.5), layout="constrained")
    axes = figure.add_subplot(projection="polar")
    axes.set_theta_zero_location("N")
    axes.set_theta_direction(-1)
    width = 2 * math.pi / 24
    shares = [100 * bin_.calling_share for bin_ in activity.by_hour if bin_.recorded_windows]
    limit = _share_limit(shares)
    for hour, bin_ in enumerate(activity.by_hour):
        angle = (hour + 0.5) * width
        if bin_.recorded_windows:
            axes.bar(angle, 100 * bin_.calling_share, width=width * 0.9, color=_CALLS)
        else:
            axes.bar(angle, limit, width=width, **_UNRECORDED)
    axes.set_ylim(0, limit)
    axes.yaxis.set_major_formatter(PercentFormatter(100))
    # The percentage labels go along the quietest hour, so they never cover a tall bar.
    quietest = min(range(24), key=lambda hour: activity.by_hour[hour].calling_share or 0.0)
    axes.set_rlabel_position(math.degrees((quietest + 0.5) * width))
    axes.set_xticks([hour * width for hour in range(0, 24, 3)], labels=[f"{hour:02d}:00" for hour in range(0, 24, 3)])
    axes.set_title(f"Calls in recorded time by hour ({activity.time_basis.label})")
    axes.legend(handles=_legend(), loc="lower center", bbox_to_anchor=(0.5, -0.12), ncols=2)
    figure.savefig(path, dpi=_DPI, format="png")


def write_by_date_figure(path: Path, activity: Activity) -> None:
    figure = Figure(figsize=(10, 5.5), layout="constrained")
    share_axes, effort_axes = figure.subplots(2, 1, sharex=True, height_ratios=(3, 1))
    days = list(activity.by_date)
    recorded = [day for day in days if activity.by_date[day].recorded_windows]
    for day in days:
        if day not in recorded:
            middle = datetime.combine(day, time())
            share_axes.axvspan(middle - timedelta(hours=12), middle + timedelta(hours=12), **_UNRECORDED)
    shares = [100 * activity.by_date[day].calling_share for day in recorded]
    share_axes.bar(recorded, shares, width=0.8, color=_CALLS)
    share_axes.set(ylim=(0, _share_limit(shares)), ylabel="Calls in recorded time")
    share_axes.yaxis.set_major_formatter(PercentFormatter(100))
    share_axes.legend(handles=_legend(), loc="upper right")
    share_axes.grid(axis="y", alpha=0.3)
    effort_axes.bar(days, [activity.by_date[day].recorded_minutes / 60 for day in days], width=0.8, color=_EFFORT)
    effort_axes.set(ylabel="Recorded hours", xlabel=f"Date ({activity.time_basis.label})")
    locator = mdates.AutoDateLocator()
    effort_axes.xaxis.set_major_locator(locator)
    effort_axes.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    if not days:
        share_axes.text(0.5, 0.5, "No recording could be placed in time", transform=share_axes.transAxes, ha="center")
    figure.suptitle(f"Calling activity by date, threshold {activity.threshold:.3f}")
    figure.savefig(path, dpi=_DPI, format="png")


def _share_limit(shares: list[float]) -> float:
    """Room above the tallest bar, and a visible scale even when every share is tiny."""
    return max(5.0, 1.15 * max(shares, default=0.0))


def _legend() -> list[Patch]:
    return [
        Patch(color=_CALLS, label="Calls in recorded time"),
        Patch(label="Not recorded", **_UNRECORDED),
    ]
