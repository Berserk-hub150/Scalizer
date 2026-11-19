"""Tests of the interfaces: sources, FileRepository, WorkloadAnalyzer, DatasetReader.

The key property: the Workload Analyzer depends only on the WorkloadSource
interface, so any source delivering the canonical raw series goes through
exactly the same cleaning, calibration and dataset steps as the Alibaba trace.
"""

import numpy as np
import pandas as pd
import pytest

from scalizer.workload import (
    CpuBasis, DatasetReader, FileRepository, Prediction, Repository, ServiceModel, ServiceModels,
    WorkloadAnalyzer, WorkloadData, WorkloadSource, validate_raw_series,
)
from scalizer.workload.__main__ import main as cli

from conftest import InMemorySource

A, B = 0.004, 0.03   # CPU model of the synthetic service: cpu = A * rps_per_pod + B


def raw_series(days=2, pods=12):
    """A canonical raw series with a daily cycle and one artefact."""
    minutes = np.arange(0, days * 1440)
    rps = 600 + 500 * np.sin(2 * np.pi * (minutes % 1440) / 1440 - np.pi / 2)
    cpu = A * rps / pods + B
    raw = pd.DataFrame({"service": "checkout", "minute": minutes, "load": rps,
                        "containers": float(pods), "cpu": cpu, "mem": np.nan,
                        "rt_ms": 80 + 20 * cpu})
    raw.loc[700, "load"] *= 4   # artefact: CPU does not follow
    return raw


def test_validate_raw_series():
    ok = pd.DataFrame({"service": ["a"], "minute": [0], "load": [1.0], "containers": [1],
                       "cpu": [0.1], "mem": [np.nan], "rt_ms": [np.nan]})
    validate_raw_series(ok)
    with pytest.raises(ValueError):
        validate_raw_series(ok.drop(columns="cpu"))
    with pytest.raises(ValueError):
        validate_raw_series(pd.concat([ok, ok]))       # same service and minute twice


# ------------------------------------------------- analyzer + repository (end to end)

def test_any_source_goes_through_the_same_pipeline(tmp_path):
    source = InMemorySource(raw_series())
    assert isinstance(source, WorkloadSource)
    repo = FileRepository(tmp_path / "datasets")
    wa = WorkloadAnalyzer(source, repo, workdir=tmp_path / "work")
    reader = DatasetReader(repo)
    assert isinstance(repo, Repository)

    ds = wa.build("checkout", None, None, version="v1")

    # Cleaning caught the artefact; calibration recovered the CPU model.
    assert ds.series.set_index("minute").loc[700, "quality"] == "spike_fixed"
    calibration = reader.get_calibration("checkout", "v1")
    cal = calibration.cpu_model
    assert cal.a == pytest.approx(A, rel=1e-3) and cal.b == pytest.approx(B, rel=1e-3)
    # The units of the source travel with the models.
    assert calibration.load_unit == "requests per second"
    assert calibration.cpu_basis is CpuBasis.FRACTION_OF_REQUEST
    assert calibration.version == "v1"

    # The reader serves the stored dataset.
    assert len(reader.get_series("checkout", "v1", start=0, end=60)) == 60
    assert set(reader.get_series("checkout", "v1", quality=("ok",))["quality"]) == {"ok"}
    assert reader.get_calibration("checkout", "v1").data_version == ds.data_version
    assert reader.list_datasets() == wa.list_datasets() == [("checkout", "v1")]


def test_engines_depend_on_the_provided_interfaces(tmp_path):
    repo = FileRepository(tmp_path / "datasets")
    WorkloadAnalyzer(InMemorySource(raw_series()), repo, workdir=tmp_path / "work").build(
        "checkout", None, None, "v1")
    reader = DatasetReader(repo)
    assert isinstance(reader, WorkloadData) and isinstance(reader, ServiceModels)
    assert isinstance(reader.get_service_model("checkout", "v1"), ServiceModel)

    # An engine written against the interfaces accepts any implementation,
    # for example a test double that never touches files.
    def peak_cpu(data: WorkloadData, models: ServiceModels, instances: int) -> float:
        peak = data.get_series("checkout", "v1")["load"].max()
        return models.get_service_model("checkout", "v1").predict(peak, instances).cpu

    class FixedModel:
        def predict(self, load, instances, params=None):
            return Prediction(load_per_instance=load / instances, cpu=0.5, rt_mean_ms=10,
                              rt_p95_ms=25, error_rate=0, saturated=False, extrapolated=False)
        predict_series = nominal_parameters = sample_parameters = instances_for = None

    class FakeModels:
        def get_calibration(self, service, version):
            raise NotImplementedError
        def get_service_model(self, service, version):
            return FixedModel()

    assert isinstance(FakeModels(), ServiceModels)
    assert peak_cpu(reader, FakeModels(), 12) == 0.5
    peak = reader.get_series("checkout", "v1")["load"].max()
    assert peak_cpu(reader, reader, 12) == pytest.approx(A * peak / 12 + B, rel=0.01)


def test_repository_versions_are_immutable(tmp_path):
    repo = FileRepository(tmp_path / "datasets")
    wa = WorkloadAnalyzer(InMemorySource(raw_series()), repo, workdir=tmp_path / "work")
    first = wa.build("checkout", None, None, "v1")
    again = wa.build("checkout", None, None, "v1")       # same data: allowed
    assert again.data_version == first.data_version
    with pytest.raises(FileExistsError):                  # other data, same version: refused
        wa.build("checkout", 0, 1440, "v1")
    wa.build("checkout", 0, 1440, "v2")
    assert repo.list() == [("checkout", "v1"), ("checkout", "v2")]


# --------------------------------------------------------------- command line

def test_command_line_with_a_saved_raw_series(tmp_path, capsys):
    raw_series().to_csv(tmp_path / "raw.csv.gz", index=False, compression="gzip")
    out = tmp_path / "work"
    cli(["clean", "--out", str(out), "--input", str(tmp_path / "raw.csv.gz")])
    cli(["build", "--out", str(out), "--input", str(tmp_path / "raw.csv.gz"),
         "--service", "checkout", "--datasets", str(tmp_path / "datasets")])
    text = capsys.readouterr().out
    assert "CPU model" in text
    ds = FileRepository(tmp_path / "datasets").load("checkout", "v1")
    assert set(ds.calibration) == {"service", "version", "data_version", "load_unit",
                                   "cpu_basis", "cpu_model", "latency_model",
                                   "p95_model", "error_model"}
    assert ds.calibration["cpu_basis"] == CpuBasis.NORMALISED_UNKNOWN_SCALE.value
    # build cleaned the data itself, with the same flags as the clean command
    clean = pd.read_csv(out / "clean" / "checkout.csv.gz")
    assert clean["quality"].tolist() == ds.series["quality"].tolist()

    # the cleaning thresholds can be overridden at build time: new data, new version
    cli(["build", "--out", str(out), "--input", str(tmp_path / "raw.csv.gz"),
         "--service", "checkout", "--version", "v2", "--max-gap", "3",
         "--datasets", str(tmp_path / "datasets")])
    v2 = FileRepository(tmp_path / "datasets").load("checkout", "v2")
    assert v2.data_version != ds.data_version
