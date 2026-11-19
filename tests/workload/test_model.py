"""Tests of features, calibration and dataset output."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scalizer.workload import (
    CalibratedServiceModel, Calibration, CpuBasis, FileRepository, add_features, calibrate, make_dataset,
)
from scalizer.workload.__main__ import main as cli

A, B = 600.0, 0.03            # true CPU model of the synthetic service
RT_BASE, RT_QUEUE = 40.0, 10.0  # true latency model


def synthetic_clean(minutes=1440, containers=100, noise=0.0, seed=0):
    """A clean series with a daily cycle that follows the true models exactly."""
    rng = np.random.default_rng(seed)
    m = np.arange(minutes)
    load = 0.03 + 0.025 * np.sin(2 * np.pi * (m % 1440) / 1440 - np.pi / 2)
    cpu = A * load / containers + B + rng.normal(0, noise, minutes)
    rt = RT_BASE + RT_QUEUE * cpu / (1 - cpu)
    return pd.DataFrame({"minute": m, "load": load, "containers": float(containers),
                         "cpu": cpu, "mem": 0.5, "rt_ms": rt, "quality": "ok"})


# ----------------------------------------------------------------------- features

def test_features():
    s = add_features(pd.DataFrame({
        "minute": [0, 61, 1500], "load": [0.1, np.nan, 0.2],
        "containers": [10.0, 10.0, 0.0], "quality": ["ok", "missing", "ok"]}))
    assert s["load_per_instance"].iloc[0] == pytest.approx(0.01)
    assert np.isnan(s["load_per_instance"].iloc[1])     # missing load -> empty
    assert np.isnan(s["load_per_instance"].iloc[2])     # no containers -> empty
    assert list(s["day"]) == [0, 0, 1]
    assert list(s["hour_of_day"]) == [0, 1, 1]


# -------------------------------------------------------------------- calibration

def test_calibration_recovers_the_true_models():
    c, fit = calibrate(add_features(synthetic_clean()))
    cpu, lat = c["cpu_model"], c["latency_model"]
    assert cpu["a"] == pytest.approx(A) and cpu["b"] == pytest.approx(B)
    assert fit["cpu_r2"] == pytest.approx(1.0)
    assert lat["rt_base"] == pytest.approx(RT_BASE) and lat["rt_queue"] == pytest.approx(RT_QUEUE)
    assert fit["cpu_test_error"] < 1e-9


def test_calibration_intervals_have_90_percent_coverage():
    # A 90% interval must miss the true value about 1 time in 10: check it over
    # many noisy series rather than on a single one.
    hits_a = hits_b = 0
    runs = 200
    for seed in range(runs):
        models, fit = calibrate(add_features(synthetic_clean(noise=0.003, seed=seed)))
        c = models["cpu_model"]
        hits_a += c["a_ci90"][0] < A < c["a_ci90"][1]
        hits_b += c["b_ci90"][0] < B < c["b_ci90"][1]
        assert 0.9 < fit["cpu_r2"] < 1.0
    assert 0.84 <= hits_a / runs <= 0.96
    assert 0.84 <= hits_b / runs <= 0.96


def test_calibration_uses_only_ok_minutes():
    s = synthetic_clean()
    s.loc[100:200, "cpu"] = 0.99                    # corrupted minutes...
    s.loc[100:200, "quality"] = "spike_fixed"       # ...flagged by cleaning
    c, fit = calibrate(add_features(s))
    assert c["cpu_model"]["a"] == pytest.approx(A)
    assert fit["minutes"] == 1440 - 101


def test_calibration_needs_enough_data():
    s = synthetic_clean(minutes=30)
    with pytest.raises(ValueError):
        calibrate(add_features(s))


def test_prediction_and_inverse():
    # The formulas live only in the service model: calibration gives the numbers.
    ds = make_dataset("T", add_features(synthetic_clean())[["minute", "load", "containers",
                                                            "cpu", "mem", "rt_ms", "quality"]],
                      source=SOURCE)
    model = CalibratedServiceModel(Calibration.model_validate(ds.calibration))
    p = model.predict(load=0.05, instances=100)
    assert p.cpu == pytest.approx(A * 0.0005 + B)
    assert p.rt_mean_ms == pytest.approx(RT_BASE + RT_QUEUE * p.cpu / (1 - p.cpu))
    n = model.instances_for(load=0.06, target_cpu=0.5)
    assert model.predict(0.06, n).cpu <= 0.5 < model.predict(0.06, n - 1).cpu
    with pytest.raises(ValueError):
        model.instances_for(0.06, target_cpu=B / 2)

# ------------------------------------------------------------------------ dataset

SOURCE = {"kind": "test", "provider": "synthetic", "load_unit": "normalised",
          "cpu_basis": "normalised_unknown_scale"}


def test_save_and_load_dataset(tmp_path):
    clean = synthetic_clean()
    params = {"spike_window": 7}
    repo = FileRepository(tmp_path)
    built = make_dataset("MS_T", clean, source=SOURCE, cleaning_params=params)
    repo.save(built)
    assert {p.name for p in Path(built.location).iterdir()} == {"series.csv.gz",
                                                                "calibration.json"}
    assert built.fit["cpu_r2"] == pytest.approx(1.0)        # computed, not stored

    ds = repo.load("MS_T", "v1")
    assert list(ds.series.columns) == ["minute", "day", "hour_of_day", "load", "containers",
                                       "load_per_instance", "cpu", "mem", "rt_ms", "quality"]
    cal = ds.calibration
    assert cal["cpu_model"]["a"] == pytest.approx(A)
    assert (cal["service"], cal["version"]) == ("MS_T", "v1")
    assert cal["load_unit"] == "normalised"
    assert ds.data_version == cal["data_version"] == built.data_version

    # same data: saving again is allowed and gives the same data_version
    repo.save(make_dataset("MS_T", clean, source=SOURCE, cleaning_params=params))
    # different cleaning choices change the data_version
    v2 = make_dataset("MS_T", clean, "v2", source=SOURCE, cleaning_params={"spike_window": 9})
    assert v2.data_version != ds.data_version
    # different data under the same version: refused
    with pytest.raises(FileExistsError):
        repo.save(make_dataset("MS_T", synthetic_clean(containers=50), source=SOURCE,
                               cleaning_params=params))


def test_load_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        FileRepository(tmp_path).load("MS_NONE", "v1")


# ---------------------------------------------------------------- command line

def test_clean_and_build_from_the_command_line(tmp_path, capsys):
    raw = pd.concat([synthetic_clean().assign(service=name) for name in ("MS_A", "MS_B")])
    raw.drop(columns="quality").to_csv(tmp_path / "raw_series.csv.gz", index=False)

    cli(["clean", "--out", str(tmp_path), "--startup-minutes", "0"])
    assert (tmp_path / "clean" / "MS_A.csv.gz").exists()
    assert "MS_B: ok 1440" in capsys.readouterr().out

    cli(["build", "--out", str(tmp_path), "--service", "MS_A",
         "--datasets", str(tmp_path / "datasets")])
    assert "CPU model" in capsys.readouterr().out
    folder = tmp_path / "datasets" / "MS_A_v1"
    assert {p.name for p in folder.iterdir()} == {"series.csv.gz", "calibration.json"}
    # the command line reads Alibaba series: their units travel with the models
    assert (json.loads((folder / "calibration.json").read_text())["cpu_basis"]
            == CpuBasis.NORMALISED_UNKNOWN_SCALE.value)
