"""The interfaces of the Workload Analyzer.

Required: what the analyzer needs from the outside world, so that the data
provider and the storage can be replaced without touching cleaning or
calibration (ports and adapters).

    WorkloadSource  gives the raw series   (today: alibaba.AlibabaTraceSource)
    Repository      stores the datasets    (today: dataset.FileRepository;
                                            later: the Database component)

Provided: what the other engines depend on, instead of the concrete classes,
so that another implementation (e.g. one reading from the Database, or a test
double) can replace them.

    WorkloadData    the clean series       Forecasting, Simulation
    ServiceModels   the calibrated models  Simulation, Optimization
    ServiceModel    the behaviour of one service: predict, sample_parameters,
                    instances_for          (today: reader.DatasetReader and
                                            model.CalibratedServiceModel)

All are Protocols: any class with these methods fits, no inheritance needed.

The raw series is the only input format. Every source turns its data into this
table: one row per service and minute.

    service     identifier of the service
    minute      integer minute from the time origin of the source
    load        inbound demand of the whole service, as a rate
    containers  number of instances serving the load
    cpu         average CPU per instance
    mem         average memory per instance (may be empty)
    rt_ms       mean response time in milliseconds (may be empty)
    rt_p95_ms   optional: p95 response time; learned instead of assumed
    error_rate  optional: share of failed requests (0-1)

Minutes without data may simply be absent: cleaning completes the time grid.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:  # type hints only
    import numpy as np

    from .contracts import (
        Calibration, CleaningParams, ModelParams, Prediction, SourceDescription,
    )
    from .dataset import Dataset

RAW_REQUIRED_COLUMNS = ("service", "minute", "load", "containers", "cpu", "mem", "rt_ms")
RAW_OPTIONAL_COLUMNS = ("rt_p95_ms", "error_rate")


# ============================================================ required interfaces

@runtime_checkable
class WorkloadSource(Protocol):
    """A provider of monitoring data."""

    def describe(self) -> "SourceDescription":
        """Units of the load and basis of the CPU: stored in calibration.json."""
        ...

    def default_cleaning(self) -> "CleaningParams":
        """Cleaning thresholds suited to this source."""
        ...

    def fetch(self, services: list[str], start: int | None, end: int | None,
              workdir: Path) -> pd.DataFrame:
        """The raw series of `services` for minutes [start, end)."""
        ...


@runtime_checkable
class Repository(Protocol):
    """Storage of versioned datasets. A version is immutable: saving different
    data under an existing version must fail."""

    def save(self, dataset: "Dataset") -> str: ...

    def load(self, service: str, version: str) -> "Dataset": ...

    def list(self) -> list[tuple[str, str]]: ...


# ============================================================ provided interfaces

@runtime_checkable
class WorkloadData(Protocol):
    """The clean per-minute series of a dataset."""

    def list_datasets(self) -> list[tuple[str, str]]: ...

    def get_series(self, service: str, version: str, start: int | None = None,
                   end: int | None = None, quality: tuple[str, ...] | None = None
                   ) -> pd.DataFrame: ...


@runtime_checkable
class ServiceModel(Protocol):
    """What a service does with a given load and number of instances."""

    def predict(self, load: float, instances: int,
                params: "ModelParams | None" = None) -> "Prediction": ...

    def predict_series(self, load, instances,
                       params: "ModelParams | None" = None) -> pd.DataFrame: ...

    def nominal_parameters(self) -> "ModelParams": ...

    def sample_parameters(self, rng: "np.random.Generator") -> "ModelParams": ...

    def instances_for(self, load: float, target_cpu: float,
                      params: "ModelParams | None" = None) -> int: ...


@runtime_checkable
class ServiceModels(Protocol):
    """The calibrated models of a dataset."""

    def get_calibration(self, service: str, version: str) -> "Calibration": ...

    def get_service_model(self, service: str, version: str) -> ServiceModel: ...


# ===================================================================== raw series

def validate_raw_series(series: pd.DataFrame) -> pd.DataFrame:
    """Check that a source respected the raw-series format; return it unchanged."""
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in series.columns]
    if missing:
        raise ValueError(f"Raw series is missing the columns {missing}; "
                         f"required: {list(RAW_REQUIRED_COLUMNS)}")
    if series.empty:
        return series
    if not pd.api.types.is_integer_dtype(series["minute"]):
        raise ValueError("'minute' must be an integer minute index")
    duplicated = series.duplicated(["service", "minute"]).sum()
    if duplicated:
        raise ValueError(f"{duplicated} rows repeat the same service and minute: "
                         "a source must deliver one row per service and minute")
    if "error_rate" in series and not series["error_rate"].dropna().between(0, 1).all():
        raise ValueError("'error_rate' must be a share between 0 and 1")
    return series
