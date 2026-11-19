"""Typed contracts and parameters of the Workload Analyzer.

    Parameters   CleaningParams, CalibrationParams: choices made before looking
                 at the data. Both enter the data_version of a dataset.
    Input        SourceDescription: who provides the data, what the numbers mean.
    Output       Calibration (the content of calibration.json), ModelParams,
                 Prediction.

The Pydantic models validate what crosses the component's boundary and give the
JSON Schema of the future REST API.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# =================================================================== parameters


@dataclass(frozen=True)
class CleaningParams:
    """Thresholds of the cleaning rules (see clean.py)."""
    spike_window: int = 7              # minutes of the rolling median used to find spikes
    spike_min_load_deviation: float = 0.20
    spike_max_cpu_ratio: float = 0.25  # CPU moves less than this × the load deviation
    collapse_threshold: float = -0.80  # load AND CPU drop more than this: collection hole
    hole_window: int = 30              # minutes before a traffic hole used as reference
    hole_max_load_share: float = 0.05  # load below 5% of the reference...
    hole_min_cpu_share: float = 0.5    # ...while CPU stays above half of it: traffic hole
    container_window: int = 3          # minutes of the rolling median for container dips
    max_interpolated_gap: int = 10     # minutes
    startup_minutes: int = 0           # Alibaba excludes 15

    def __post_init__(self):
        problems = []
        if self.spike_window < 3:
            problems.append("spike_window must be at least 3")
        if self.spike_min_load_deviation <= 0:
            problems.append("spike_min_load_deviation must be positive")
        if self.spike_max_cpu_ratio < 0:
            problems.append("spike_max_cpu_ratio must not be negative")
        if not -1 < self.collapse_threshold < 0:
            problems.append("collapse_threshold must be between -1 and 0 (a drop)")
        if self.hole_window < 1:
            problems.append("hole_window must be positive")
        if not 0 < self.hole_max_load_share < 1 or not 0 < self.hole_min_cpu_share <= 1:
            problems.append("hole_max_load_share and hole_min_cpu_share must be shares")
        if self.container_window < 1 or self.container_window % 2 == 0:
            problems.append("container_window must be a positive odd number")
        if self.max_interpolated_gap < 0 or self.startup_minutes < 0:
            problems.append("max_interpolated_gap and startup_minutes must not be negative")
        if problems:
            raise ValueError("invalid cleaning parameters: " + "; ".join(problems))


@dataclass(frozen=True)
class CalibrationParams:
    train_fraction: float = 0.5      # first part of the ok minutes, to test on the rest
    min_minutes: int = 60            # fewer ok minutes: refuse to calibrate
    p95_factor: float = 2.5          # assumption when the source has no p95:
    p95_factor_range: tuple = (2.0, 3.0)   # p95 = k × mean
    saturation_cpu: float = 0.9      # assumption: errors grow above this CPU
    assumed_base_error_rate: float = 0.0   # assumption when the source has no errors


# ===================================================================== contracts

class _Model(BaseModel):
    # Unknown keys are kept, so that older datasets still load.
    model_config = ConfigDict(extra="allow", frozen=True)


class CpuBasis(str, Enum):
    """What the `cpu` column is a fraction of. A threshold such as "scale out at
    0.7" is only meaningful in the same basis as the data."""
    FRACTION_OF_REQUEST = "fraction_of_request"   # Kubernetes HPA convention
    FRACTION_OF_LIMIT = "fraction_of_limit"
    FRACTION_OF_CORE = "fraction_of_core"
    FRACTION_OF_VM = "fraction_of_vm"
    NORMALISED_UNKNOWN_SCALE = "normalised_unknown_scale"   # the Alibaba trace


class SourceDescription(_Model):
    kind: str = Field(description="Type of source, e.g. 'alibaba-microservices-v2022'")
    provider: str
    location: str | None = None
    load_unit: str = Field(description="Unit of the load column")
    cpu_basis: CpuBasis
    time_step_seconds: int = 60
    time_origin: str | None = Field(None, description="ISO date-time of minute 0, if known")
    notes: str = ""


Interval = tuple[float, float]   # a 90% interval or an observed range, [low, high]


class CpuModel(_Model):
    """cpu = a × load_per_instance + b"""
    a: float
    a_ci90: Interval
    b: float
    b_ci90: Interval
    a_b_correlation: float = 0.0        # a and b are estimated together: drawn together
    load_per_instance_range: Interval   # outside it: an extrapolation


class LatencyModel(_Model):
    """rt_ms = rt_base + rt_queue × cpu / (1 − cpu)"""
    rt_base: float
    rt_base_ci90: Interval
    rt_queue: float
    rt_queue_ci90: Interval
    rt_base_rt_queue_correlation: float = 0.0
    observed_cpu_max: float             # above it: an extrapolation


class P95Model(_Model):
    """Observed: same form as the latency model, fitted on rt_p95_ms.
    Factor: p95 = k × mean, k an assumption."""
    method: Literal["observed", "factor"]
    p95_base: float | None = None
    p95_base_ci90: Interval | None = None
    p95_queue: float | None = None
    p95_queue_ci90: Interval | None = None
    p95_base_p95_queue_correlation: float = 0.0
    k: float | None = None
    k_range: Interval | None = None
    assumption: bool


class ErrorModel(_Model):
    """Base rate below saturation_cpu, then linear growth to 100% at CPU 1."""
    method: Literal["observed", "assumed"]
    base_error_rate: float
    saturation_cpu: float
    assumption: bool


class Calibration(_Model):
    """Content of calibration.json."""
    service: str
    version: str
    data_version: str
    load_unit: str
    cpu_basis: CpuBasis
    cpu_model: CpuModel
    latency_model: LatencyModel
    p95_model: P95Model
    error_model: ErrorModel


class ModelParams(_Model):
    """One set of parameters: the nominal one, or a sample for a repeated run."""
    a: float
    b: float
    rt_base: float
    rt_queue: float
    p95_base: float | None = None
    p95_queue: float | None = None
    p95_k: float | None = None
    base_error_rate: float
    saturation_cpu: float


class Prediction(_Model):
    """What the service does with a given load and number of instances."""
    load_per_instance: float
    cpu: float
    rt_mean_ms: float = Field(description="inf when the service is saturated")
    rt_p95_ms: float
    error_rate: float
    saturated: bool = Field(description="cpu >= 1: demand exceeds capacity")
    extrapolated: bool = Field(description="outside the conditions observed in the data")
