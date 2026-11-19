"""Cleaning: one row per minute, artefacts corrected, a quality flag per minute.

    clean_series(raw, start, end, params)

The steps run in order; each one corrects the table and records the minutes it
touched. Only "ok" minutes are used to calibrate.

    ok                measured and unchanged
    containers_fixed  one-minute dip in the number of containers, smoothed
    spike_fixed       load spike that the CPU did not follow, corrected
    interpolated      gap of up to max_interpolated_gap minutes, filled
    missing           longer gap, or start-up transient: no traffic values

When two steps touch the same minute, the stronger flag (later in the list) wins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import CleaningParams
from .interfaces import RAW_OPTIONAL_COLUMNS

MEASURED = ("load", "containers", "cpu", "mem", "rt_ms")
RESOURCES = ("containers", "cpu", "mem")
QUALITY_FLAGS = ("ok", "containers_fixed", "spike_fixed", "interpolated", "missing")


def clean_series(raw: pd.DataFrame, start: int, end: int,
                 params: CleaningParams = CleaningParams()) -> pd.DataFrame:
    """Clean one service on [start, end). The result has one row per minute,
    with `quality` as the last column."""
    if "service" in raw and raw["service"].nunique() > 1:
        raise ValueError("clean_series works on one service at a time")
    if end <= start:
        raise ValueError(f"empty period: start={start}, end={end}")
    optional = [c for c in RAW_OPTIONAL_COLUMNS if c in raw and raw[c].notna().any()]
    traffic = ["load", "rt_ms", *optional]   # what the service received and answered

    s = complete_time_grid(raw, start, end, optional)
    gap = blank_gaps(s, traffic)
    gap |= blank_traffic_holes(s, traffic, params)
    containers_fixed = smooth_container_dips(s, params.container_window)
    spike, collapse = fix_load_spikes(s, traffic, params)
    gap |= collapse
    interpolated = interpolate_short_gaps(s, gap, params.max_interpolated_gap)
    startup = exclude_startup(s, traffic, params.startup_minutes)

    # Weakest first: a stronger flag overwrites a weaker one.
    quality = pd.Series("ok", index=s.index, dtype=object)
    quality[containers_fixed] = "containers_fixed"
    quality[spike] = "spike_fixed"
    quality[interpolated] = "interpolated"
    quality[(gap & ~interpolated) | startup] = "missing"
    return s.assign(quality=quality).reset_index()


# 1. ------------------------------------------------------------------------------

def complete_time_grid(raw, start, end, optional) -> pd.DataFrame:
    """One row per minute of [start, end); absent minutes become empty rows."""
    missing = [c for c in ("minute", *MEASURED) if c not in raw]
    if missing:
        raise ValueError(f"raw series is missing the columns {missing}")
    minutes = pd.RangeIndex(start, end, name="minute")
    observed = raw.drop_duplicates("minute").set_index("minute")
    return observed[[*MEASURED, *optional]].reindex(minutes).astype(float)


# 2. ------------------------------------------------------------------------------

def blank_gaps(s, traffic) -> pd.Series:
    """Minutes with no traffic data: no load, or zero load with no latency while
    the CPU was measured. Their traffic is blanked; CPU and containers stay."""
    gap = s["load"].isna() | ((s["load"] <= 0) & s["rt_ms"].isna() & s["cpu"].notna())
    s.loc[gap, traffic] = np.nan
    return gap


def blank_traffic_holes(s, traffic, params) -> pd.Series:
    """Hours in which the recorded load collapses while the service keeps working.

    The load falls below hole_max_load_share of its median over the previous
    hole_window minutes, while the CPU stays above hole_min_cpu_share of its own.
    The inbound calls are not measured there (in the Alibaba trace, a residual
    rate of about 0.00007 and a latency of about 1 ms remain), so these minutes
    would read as "no load, normal CPU" and distort the CPU model. A hole lasts
    while the load stays at most twice its level at the start of the hole.
    """
    before = s[["load", "cpu"]].rolling(params.hole_window, min_periods=1).median().shift(1)
    starts = ((s["load"] < params.hole_max_load_share * before["load"])
              & (s["cpu"] >= params.hole_min_cpu_share * before["cpu"]))
    hole = pd.Series(False, index=s.index)
    level = None                               # load at the start of the current hole
    for i, (load, is_start) in enumerate(zip(s["load"], starts)):
        if level is None and is_start:
            level = load
        elif level is not None and not load <= 2 * max(level, 0):
            level = None                       # the load came back: the hole is over
        hole.iat[i] = level is not None
    s.loc[hole, traffic] = np.nan
    return hole


# 3. ------------------------------------------------------------------------------

def smooth_container_dips(s, window) -> pd.Series:
    """Counts that differ from the rounded median of `window` minutes are replaced."""
    median = s["containers"].rolling(window, center=True, min_periods=1).median().round()
    fixed = s["containers"].notna() & (s["containers"] != median)
    s.loc[fixed, "containers"] = median[fixed]
    return fixed


# 4. ------------------------------------------------------------------------------

def fix_load_spikes(s, traffic, params) -> tuple[pd.Series, pd.Series]:
    """Compare load and CPU with their median of `spike_window` minutes.

    spike     load moves by more than spike_min_load_deviation and the CPU moves
              much less: an artefact, the load is replaced by its median.
    collapse  load AND CPU both drop by more than collapse_threshold: a hole in
              the collection, every measure of the minute is blanked.
    Large changes that the CPU follows are real traffic and are kept.
    """
    load_dev, load_median = relative_deviation(s["load"], params.spike_window)
    cpu_dev, _ = relative_deviation(s["cpu"], params.spike_window)
    large = load_dev.abs() > params.spike_min_load_deviation
    collapse = (large & (load_dev < params.collapse_threshold)
                & (cpu_dev < params.collapse_threshold))
    weak_cpu = cpu_dev.abs().fillna(0) < params.spike_max_cpu_ratio * load_dev.abs()
    spike = large & ~collapse & weak_cpu
    s.loc[spike, "load"] = load_median[spike]
    s.loc[collapse, [*traffic, *RESOURCES]] = np.nan
    return spike, collapse


def relative_deviation(values, window) -> tuple[pd.Series, pd.Series]:
    """Deviation from the centred median, as a fraction: +2.0 = three times the
    median. The median, unlike the mean, is not pulled by the spike itself."""
    median = values.rolling(window, center=True, min_periods=3).median()
    return (values - median) / median.where(median > 0), median


# 5. ------------------------------------------------------------------------------

def interpolate_short_gaps(s, gap, max_gap) -> pd.Series:
    """Gaps of at most max_gap minutes, with data on both sides, are filled with
    a straight line. Returns the minutes filled."""
    run = (gap != gap.shift()).cumsum()                 # one id per run of equal values
    minutes = pd.Series(s.index, index=s.index)
    first = minutes.groupby(run).transform("min")
    last = minutes.groupby(run).transform("max")
    short = (gap & (last - first + 1 <= max_gap)
             & (first > s.index[0]) & (last < s.index[-1]))
    if short.any():
        line = s.interpolate(method="index", limit_area="inside")
        s.loc[short, :] = line.loc[short, :]
    s["containers"] = s["containers"].round()
    return short


# 6. ------------------------------------------------------------------------------

def exclude_startup(s, traffic, startup_minutes) -> pd.Series:
    """The first minutes of the data (absolute minutes) are unstable: traffic blanked."""
    startup = pd.Series(s.index < startup_minutes, index=s.index)
    s.loc[startup, traffic] = np.nan
    return startup
