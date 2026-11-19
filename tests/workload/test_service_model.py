"""Tests of the behavioural ServiceModel and of the optional input columns."""
import json

import numpy as np
import pandas as pd
import pytest

from scalizer.workload import (
    Calibration, FileRepository, Prediction, ServiceModel,
    DatasetReader, WorkloadAnalyzer,
)
from conftest import InMemorySource

A, B, RT0, RTQ = 0.004, 0.03, 80.0, 20.0     # the true models of the synthetic service


def raw_series(with_p95=False, with_errors=False, pods=12, noise=0.0):
    minutes = np.arange(0, 2 * 1440)
    rps = 600 + 500 * np.sin(2 * np.pi * (minutes % 1440) / 1440 - np.pi / 2)
    cpu = A * rps / pods + B + np.random.default_rng(1).normal(0, noise, len(minutes))
    raw = pd.DataFrame({"service": "api", "minute": minutes, "load": rps,
                        "containers": float(pods), "cpu": cpu, "mem": np.nan,
                        "rt_ms": RT0 + RTQ * cpu / (1 - cpu)})
    if with_p95:
        raw["rt_p95_ms"] = 150 + 60 * cpu / (1 - cpu)
    if with_errors:
        raw["error_rate"] = 0.002
    return raw


def analyzer(tmp_path, **kwargs):
    repo = FileRepository(tmp_path / "ds")
    WorkloadAnalyzer(InMemorySource(raw_series(**kwargs)), repo,
                     workdir=tmp_path / "w").build("api", None, None, "v1")
    return DatasetReader(repo)


@pytest.fixture
def model(tmp_path):
    return analyzer(tmp_path).get_service_model("api", "v1")


def test_model_implements_the_interface(model):
    assert isinstance(model, ServiceModel)


def test_predict_follows_the_calibrated_laws(model):
    p = model.predict(load=600, instances=12)
    cpu = A * 600 / 12 + B
    assert isinstance(p, Prediction)
    assert p.cpu == pytest.approx(cpu, rel=1e-3)
    assert p.rt_mean_ms == pytest.approx(RT0 + RTQ * cpu / (1 - cpu), rel=1e-3)
    assert p.rt_p95_ms == pytest.approx(2.5 * p.rt_mean_ms)     # no p95 in the data: factor
    assert p.error_rate == 0 and not p.saturated and not p.extrapolated


def test_fewer_instances_extrapolate_then_saturate(model):
    busy = model.predict(load=1100, instances=6)      # load per instance never observed
    assert busy.extrapolated and not busy.saturated
    overloaded = model.predict(load=1100, instances=4)
    assert overloaded.saturated and overloaded.rt_mean_ms == float("inf")
    assert overloaded.error_rate == 1.0


def test_predict_series_is_vectorised_and_skips_missing_load(model):
    out = model.predict_series(np.array([300, np.nan, 900]), 12)
    assert len(out) == 3 and np.isnan(out.loc[1, "cpu"])
    assert out.loc[2, "cpu"] > out.loc[0, "cpu"]


def test_sampled_parameters_vary_around_the_nominal_ones(tmp_path):
    # Noisy data: the fitted parameters have real uncertainty to sample from.
    model = analyzer(tmp_path, noise=0.003).get_service_model("api", "v1")
    rng = np.random.default_rng(0)
    nominal = model.nominal_parameters()
    samples = [model.sample_parameters(rng) for _ in range(500)]
    assert np.mean([s.a for s in samples]) == pytest.approx(nominal.a, rel=1e-2)
    assert len({s.a for s in samples}) > 1                      # really random
    assert all(2.0 <= s.p95_k <= 3.0 for s in samples)          # assumed range
    p = model.predict(600, 12, params=samples[0])
    assert p.cpu == pytest.approx(samples[0].a * 50 + samples[0].b)
    assert model.predict(600, 12).cpu != pytest.approx(p.cpu, rel=1e-9)


def test_instances_for_a_target(model):
    n = model.instances_for(load=1100, target_cpu=0.5)
    assert model.predict(1100, n).cpu <= 0.5 < model.predict(1100, n - 1).cpu
    with pytest.raises(ValueError):
        model.instances_for(1100, target_cpu=B / 2)              # below idle CPU


def test_p95_and_errors_learned_when_the_source_has_them(tmp_path):
    wa = analyzer(tmp_path, with_p95=True, with_errors=True)
    cal = wa.get_calibration("api", "v1")
    assert cal.p95_model.method == "observed" and not cal.p95_model.assumption
    assert cal.p95_model.p95_base == pytest.approx(150, rel=1e-3)
    assert cal.error_model.method == "observed"
    assert cal.error_model.base_error_rate == pytest.approx(0.002)
    p = wa.get_service_model("api", "v1").predict(600, 12)
    u = (A * 50 + B) / (1 - (A * 50 + B))
    assert p.rt_p95_ms == pytest.approx(150 + 60 * u, rel=1e-3)
    assert p.error_rate == pytest.approx(0.002)
    assert "rt_p95_ms" in wa.get_series("api", "v1").columns


def test_contracts_are_serialisable_for_the_api(tmp_path):
    wa = analyzer(tmp_path)
    cal = wa.get_calibration("api", "v1")
    again = Calibration.model_validate_json(cal.model_dump_json())
    assert again.model_dump(mode="json") == cal.model_dump(mode="json")
    schema = Prediction.model_json_schema()                  # used for the OpenAPI contract
    assert {"cpu", "rt_p95_ms", "extrapolated"} <= set(schema["properties"])
    assert cal.cpu_basis.value == "fraction_of_request"
    json.dumps(cal.model_dump(mode="json"))
