"""The calibrated model of a service, as behaviour (interfaces.ServiceModel).

The Simulation and Optimization Engines ask this object what happens with a
given load and number of instances; they never re-implement the formulas.

    model = reader.get_service_model("MS_70785", "v1")
    p = model.predict(load=0.0635, instances=85)
    p.cpu, p.rt_mean_ms, p.rt_p95_ms, p.error_rate, p.saturated, p.extrapolated

    params = model.sample_parameters(numpy.random.default_rng(42))   # one repeated run
    model.predict_series(loads, instances, params)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .calibrate import Z90, queue_term
from .contracts import Calibration, ModelParams, Prediction

RANGE_TOLERANCE = 0.05   # slightly outside the observed range is not an extrapolation


def sd_from_ci90(ci) -> float:
    """Standard deviation implied by a 90% interval [low, high] of a normal distribution."""
    low, high = ci
    return (high - low) / (2 * Z90)


def draw_pair(rng: np.random.Generator, first, first_ci, second, second_ci,
              correlation) -> tuple[float, float]:
    """Draw two parameters from a bivariate normal distribution: each centred on
    its value, with the spread of its 90% interval and the given correlation."""
    z1, z2 = rng.standard_normal(2)
    rho = min(max(correlation, -1.0), 1.0)
    return (float(first + sd_from_ci90(first_ci) * z1),
            float(second + sd_from_ci90(second_ci) * (rho * z1 + math.sqrt(1 - rho ** 2) * z2)))


class CalibratedServiceModel:
    """interfaces.ServiceModel, built from calibration.json."""

    def __init__(self, calibration: Calibration):
        self.calibration = c = calibration
        low, high = c.cpu_model.load_per_instance_range
        self._lpi_min = low * (1 - RANGE_TOLERANCE)
        self._lpi_max = high * (1 + RANGE_TOLERANCE)
        self._cpu_max = c.latency_model.observed_cpu_max * (1 + RANGE_TOLERANCE)
        self._p95_observed = c.p95_model.method == "observed"

    # ------------------------------------------------------------ parameters

    def nominal_parameters(self) -> ModelParams:
        """The fitted parameters."""
        c = self.calibration
        return ModelParams(
            a=c.cpu_model.a, b=c.cpu_model.b,
            rt_base=c.latency_model.rt_base, rt_queue=c.latency_model.rt_queue,
            p95_base=c.p95_model.p95_base, p95_queue=c.p95_model.p95_queue,
            p95_k=None if self._p95_observed else c.p95_model.k,
            base_error_rate=c.error_model.base_error_rate,
            saturation_cpu=c.error_model.saturation_cpu)

    def sample_parameters(self, rng: np.random.Generator) -> ModelParams:
        """Parameters drawn within their 90% intervals, for one repeated run.

        The two parameters of each model (a and b; rt_base and rt_queue; p95
        base and queue) are estimated together and are correlated, so they are
        drawn together from a bivariate normal distribution. The order of the
        draws is fixed (CPU, latency, p95): the same seed gives the same
        parameters. Error parameters are assumptions and are not sampled."""
        c = self.calibration
        a, b = draw_pair(rng, c.cpu_model.a, c.cpu_model.a_ci90, c.cpu_model.b,
                         c.cpu_model.b_ci90, c.cpu_model.a_b_correlation)
        lat = c.latency_model
        rt_base, rt_queue = draw_pair(rng, lat.rt_base, lat.rt_base_ci90, lat.rt_queue,
                                      lat.rt_queue_ci90, lat.rt_base_rt_queue_correlation)
        updates = {"a": a, "b": b, "rt_base": rt_base,
                   "rt_queue": max(0.0, rt_queue)}       # the queue effect cannot be negative
        p95 = c.p95_model
        if self._p95_observed:
            base, queue = draw_pair(rng, p95.p95_base, p95.p95_base_ci90, p95.p95_queue,
                                    p95.p95_queue_ci90, p95.p95_base_p95_queue_correlation)
            updates["p95_base"], updates["p95_queue"] = base, max(0.0, queue)
        else:
            updates["p95_k"] = float(rng.uniform(*p95.k_range))
        return self.nominal_parameters().model_copy(update=updates)

    # ------------------------------------------------------------ behaviour

    def predict(self, load: float, instances: int,
                params: ModelParams | None = None) -> Prediction:
        """Prediction for one load and number of instances."""
        row = self.predict_series(np.array([load], float), np.array([instances], float),
                                  params).iloc[0]
        return Prediction(**{k: (bool(v) if k in ("saturated", "extrapolated") else float(v))
                             for k, v in row.items()})

    def predict_series(self, load, instances, params: ModelParams | None = None
                       ) -> pd.DataFrame:
        """One row per element of `load` and `instances`. Missing load (NaN)
        gives NaN outputs: the simulation skips those minutes."""
        p = params or self.nominal_parameters()
        load = np.asarray(load, float)
        instances = np.broadcast_to(np.asarray(instances, float), load.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            lpi = load / np.where(instances > 0, instances, np.nan)
            cpu = p.a * lpi + p.b
            saturated = cpu >= 1
            u = np.where(saturated, np.inf, queue_term(cpu))
            rt_mean = p.rt_base + p.rt_queue * u
            rt_p95 = (p.p95_base + p.p95_queue * u) if self._p95_observed else p.p95_k * rt_mean
            # errors: base rate, then linear growth to 100% at cpu = 1
            growth = np.clip((cpu - p.saturation_cpu) / (1 - p.saturation_cpu), 0, 1)
            error = p.base_error_rate + (1 - p.base_error_rate) * growth
        extrapolated = (lpi > self._lpi_max) | (lpi < self._lpi_min) | (cpu > self._cpu_max)
        out = pd.DataFrame({
            "load_per_instance": lpi, "cpu": cpu, "rt_mean_ms": rt_mean, "rt_p95_ms": rt_p95,
            "error_rate": error, "saturated": saturated, "extrapolated": extrapolated})
        missing = np.isnan(load)
        out.loc[missing, ["cpu", "rt_mean_ms", "rt_p95_ms", "error_rate"]] = np.nan
        out.loc[missing, ["saturated", "extrapolated"]] = False
        return out

    def instances_for(self, load: float, target_cpu: float,
                      params: ModelParams | None = None) -> int:
        """Fewest instances keeping the CPU at or below target_cpu: the CPU model
        read backwards, ceil(load × a / (target − b))."""
        p = params or self.nominal_parameters()
        if target_cpu <= p.b:
            raise ValueError(f"Target CPU {target_cpu} is below the idle CPU of an "
                             f"instance ({p.b:.3f}): no number of instances reaches it")
        return max(1, math.ceil(load * p.a / (target_cpu - p.b)))
