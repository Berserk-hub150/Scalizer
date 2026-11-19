import numpy as np
import pandas as pd
import pytest

from scalizer.workload import CleaningParams, clean_series

START, END = 100, 200   # after the start-up transient


def base_series(start=START, end=END):
    """A steady service: 10 containers, load 0.05, CPU proportional to load."""
    minutes = np.arange(start, end)
    load = np.full(len(minutes), 0.05)
    return pd.DataFrame({
        "minute": minutes, "load": load, "containers": 10.0,
        "cpu": 0.02 + 4.0 * load, "mem": 0.5, "rt_ms": 40.0,
    })


def run(df, start=START, end=END, **params):
    clean = clean_series(df, start, end, CleaningParams(**params))
    return clean.set_index("minute")


def test_clean_series_unchanged():
    clean = run(base_series())
    assert len(clean) == END - START
    assert (clean["quality"] == "ok").all()
    assert clean["load"].eq(0.05).all()


def test_load_spike_without_cpu_is_an_artefact():
    df = base_series()
    df.loc[df.minute == 150, "load"] = 0.15                # load x3, CPU unchanged
    clean = run(df)
    assert clean.loc[150, "quality"] == "spike_fixed"
    assert clean.loc[150, "load"] == pytest.approx(0.05)
    assert (clean.drop(index=150)["quality"] == "ok").all()     # only that minute


def test_real_peak_with_cpu_is_kept():
    df = base_series()
    peak = df.minute.between(150, 152)
    df.loc[peak, "load"] = 0.15
    df.loc[peak, "cpu"] = 0.02 + 4.0 * 0.15                # CPU follows the load
    clean = run(df)
    assert (clean.loc[150:152, "quality"] == "ok").all()
    assert clean.loc[150:152, "load"].tolist() == pytest.approx([0.15] * 3)   # unchanged


def test_zero_load_without_response_time_is_a_gap_not_zero_traffic():
    df = base_series()
    hole = df.minute.between(130, 159)                     # 30 minutes, CPU still reported
    df.loc[hole, "load"] = 0.0
    df.loc[hole, "rt_ms"] = np.nan
    clean = run(df)
    assert (clean.loc[130:159, "quality"] == "missing").all()
    assert clean.loc[130:159, "load"].isna().all()         # empty, never 0
    assert clean.loc[130:159, "cpu"].notna().all()         # CPU was measured: kept
    assert (clean.loc[129, "quality"], clean.loc[160, "quality"]) == ("ok", "ok")


def test_short_gap_is_interpolated_long_gap_is_not():
    df = base_series()
    df.loc[df.minute.between(120, 122), "load"] = np.nan   # rows lost: 3 minutes
    df.loc[df.minute.between(120, 122), "cpu"] = np.nan
    df = df[~df.minute.between(160, 179)]                  # 20 minutes absent
    clean = run(df)
    assert (clean.loc[120:122, "quality"] == "interpolated").all()
    assert clean.loc[121, "load"] == pytest.approx(0.05)
    assert (clean.loc[160:179, "quality"] == "missing").all()   # the absent rows exist
    assert clean.loc[160:179, "load"].isna().all()


def test_one_minute_container_dip_is_smoothed_real_scaling_is_kept():
    df = base_series()
    df.loc[df.minute == 130, "containers"] = 7             # dip for one minute
    df.loc[df.minute >= 170, "containers"] = 14            # real scale-out
    clean = run(df)
    assert clean.loc[130, "containers"] == 10
    assert clean.loc[130, "quality"] == "containers_fixed"
    assert clean.loc[175, "containers"] == 14
    assert clean.loc[175, "quality"] == "ok"


def test_collapse_of_load_and_cpu_is_a_collection_hole():
    df = base_series()
    df.loc[df.minute == 140, ["load", "cpu"]] = [0.004, 0.003]
    clean = run(df)
    assert clean.loc[140, "quality"] == "interpolated"     # a hole of one minute: filled
    assert clean.loc[140, "load"] == pytest.approx(0.05)


def test_traffic_hole_with_normal_cpu_is_missing():
    # As in the Alibaba trace: for an hour the recorded load drops to a tiny
    # residual and the latency to ~1 ms, while CPU and containers stay normal.
    minutes = np.arange(0, 300)
    df = base_series(0, 300)
    hole = (minutes >= 120) & (minutes < 180)
    df.loc[hole, "load"] = 0.00007
    df.loc[hole, "rt_ms"] = 1.0
    clean = run(df, 0, 300)
    assert (clean.loc[120:179, "quality"] == "missing").all()
    assert clean.loc[120:179, "load"].isna().all()
    assert clean.loc[120:179, "cpu"].notna().all()         # resources are real measures
    assert (clean.drop(index=range(120, 180))["quality"] == "ok").all()


def test_short_traffic_hole_is_interpolated():
    df = base_series()
    df.loc[df.minute.between(150, 154), ["load", "rt_ms"]] = [0.0, 1.0]
    clean = run(df)
    assert (clean.loc[150:154, "quality"] == "interpolated").all()
    assert clean.loc[150:154, "load"].to_numpy() == pytest.approx(0.05)


def test_real_drop_of_traffic_is_not_a_hole():
    # Load and CPU fall together: the service really has less traffic.
    df = base_series()
    later = df.minute >= 150
    df.loc[later, "load"] = 0.001
    df.loc[later, "cpu"] = 0.02 + 4.0 * 0.001
    clean = run(df)
    assert clean.loc[160:199, "load"].eq(0.001).all()
    assert (clean.loc[160:199, "quality"] == "ok").all()


def test_startup_transient_is_excluded():
    df = base_series(0, 40)
    clean = run(df, start=0, end=40, startup_minutes=15)
    assert (clean.loc[0:14, "quality"] == "missing").all()
    assert clean.loc[0:14, "load"].isna().all()
    assert clean.loc[0:14, "cpu"].notna().all()            # resources are kept
    assert (clean.loc[15:, "quality"] == "ok").all()


def test_gap_at_the_edge_cannot_be_interpolated():
    df = base_series()
    df = df[df.minute >= 103]                              # first 3 minutes absent
    clean = run(df)
    assert (clean.loc[100:102, "quality"] == "missing").all()


def test_no_startup_exclusion_by_default():
    # The start-up transient is specific to the Alibaba trace: generic data keeps
    # its first minutes unless the source says otherwise.
    clean = run(base_series(0, 40), start=0, end=40)
    assert (clean["quality"] == "ok").all()


def test_gap_longer_than_the_threshold_stays_missing():
    df = base_series()
    df = df[~df.minute.between(120, 125)]                  # 6 minutes absent
    assert (run(df)["quality"].loc[120:125] == "interpolated").all()
    assert (run(df, max_interpolated_gap=5)["quality"].loc[120:125] == "missing").all()


def test_one_service_at_a_time():
    df = pd.concat([base_series().assign(service="A"), base_series().assign(service="B")])
    with pytest.raises(ValueError):
        clean_series(df, START, END)


@pytest.mark.parametrize("params", [
    {"spike_window": 2},
    {"container_window": 4},
    {"collapse_threshold": 0.5},
    {"max_interpolated_gap": -1},
    {"startup_minutes": -5},
])
def test_invalid_parameters_are_rejected(params):
    with pytest.raises(ValueError, match="invalid cleaning parameters"):
        CleaningParams(**params)


def test_missing_columns_are_reported():
    with pytest.raises(ValueError, match="rt_ms"):
        clean_series(base_series().drop(columns="rt_ms"), START, END)


def test_empty_period_is_rejected():
    with pytest.raises(ValueError, match="empty period"):
        clean_series(base_series(), END, START)


def test_strongest_flag_wins():
    # Minute 15 has a container dip, but it is also in the start-up transient:
    # "missing" is stronger than "containers_fixed".
    df = base_series(0, 40)
    df.loc[df.minute == 10, "containers"] = 7
    clean = run(df, start=0, end=40, startup_minutes=15)
    assert clean.loc[10, "quality"] == "missing"
    assert clean.loc[10, "containers"] == 10          # the value is still corrected
