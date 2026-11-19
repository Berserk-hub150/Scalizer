"""Calibration: the models of the service, fitted on the "ok" minutes.

    CPU      cpu   = a × load_per_instance + b
    latency  rt_ms = rt_base + rt_queue × cpu / (1 − cpu)
    p95      observed (same form as latency), or k × mean (assumed)
    errors   base rate, growing above the saturation CPU (assumed)

The form of each model is our choice; the parameters are learned from the data,
except those marked as assumptions. calibrate() returns the blocks stored in
calibration.json and, separately, the quality of the fit (printed, not stored).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import CalibrationParams

# Forms of the models: part of the data_version.
CPU_FORMULA = "cpu = a * load_per_instance + b"
LATENCY_FORMULA = "rt_ms = rt_base + rt_queue * cpu / (1 - cpu)"
Z90 = 1.645   # two-sided 90% interval of a normal distribution


def add_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Columns derived from the clean series.

    load_per_instance  load / containers: the CPU of an instance depends on its
                       share of the traffic, not on the total.
    day, hour_of_day   position in the daily cycle (hour 0 = first hour of the data).
    """
    out = clean.copy()
    out["load_per_instance"] = out["load"] / out["containers"].where(out["containers"] > 0)
    out["day"] = out["minute"] // 1440
    out["hour_of_day"] = (out["minute"] % 1440) // 60
    return out


def queue_term(cpu):
    """cpu / (1 − cpu): close to cpu when idle, unbounded as cpu approaches 1."""
    return cpu / (1 - cpu)


def fit_linear(x: np.ndarray, y: np.ndarray) -> dict:
    """Least squares y = slope × x + intercept, with 90% intervals from the
    standard errors se = sqrt(diag(σ² (XᵀX)⁻¹)), σ² = SSR / (n − 2)."""
    X = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coef
    sigma2 = residuals @ residuals / max(len(y) - 2, 1)
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
    total = ((y - y.mean()) ** 2).sum()
    return {
        "slope": float(coef[0]),
        "slope_ci90": [float(coef[0] - Z90 * se[0]), float(coef[0] + Z90 * se[0])],
        "intercept": float(coef[1]),
        "intercept_ci90": [float(coef[1] - Z90 * se[1]), float(coef[1] + Z90 * se[1])],
        "r2": float(1 - residuals @ residuals / total) if total > 0 else None,
    }


def mape(predicted, observed) -> float:
    """Mean absolute percentage error, as a fraction (0.024 = 2.4%)."""
    return float(np.mean(np.abs(predicted - observed) / np.abs(observed)))


# ------------------------------------------------------------------- the models

def fit_cpu(ok: pd.DataFrame) -> tuple[dict, float]:
    lpi = ok["load_per_instance"].to_numpy()
    fit = fit_linear(lpi, ok["cpu"].to_numpy())
    block = {"a": fit["slope"], "a_ci90": fit["slope_ci90"],
             "b": fit["intercept"], "b_ci90": fit["intercept_ci90"],
             "load_per_instance_range": [float(lpi.min()), float(lpi.max())]}
    return block, fit["r2"]


def fit_latency(lat: pd.DataFrame) -> dict:
    cpu = lat["cpu"].to_numpy()
    fit = fit_linear(queue_term(cpu), lat["rt_ms"].to_numpy())
    return {"rt_base": fit["intercept"], "rt_base_ci90": fit["intercept_ci90"],
            "rt_queue": fit["slope"], "rt_queue_ci90": fit["slope_ci90"],
            "observed_cpu_max": float(cpu.max())}


def fit_p95(lat: pd.DataFrame, params: CalibrationParams) -> dict:
    """Learned from rt_p95_ms when the source has enough of it, otherwise assumed."""
    if "rt_p95_ms" in lat and lat["rt_p95_ms"].notna().sum() >= params.min_minutes:
        d = lat[lat["rt_p95_ms"].notna()]
        fit = fit_linear(queue_term(d["cpu"].to_numpy()), d["rt_p95_ms"].to_numpy())
        return {"method": "observed",
                "p95_base": fit["intercept"], "p95_base_ci90": fit["intercept_ci90"],
                "p95_queue": fit["slope"], "p95_queue_ci90": fit["slope_ci90"],
                "assumption": False}
    return {"method": "factor", "k": params.p95_factor,
            "k_range": list(params.p95_factor_range), "assumption": True}


def fit_errors(ok: pd.DataFrame, params: CalibrationParams) -> dict:
    """Base rate observed when the source has errors; growth above saturation
    is always an assumption."""
    if "error_rate" in ok and ok["error_rate"].notna().sum() >= params.min_minutes:
        base = float(ok.loc[ok["cpu"] < params.saturation_cpu, "error_rate"].median())
        method = "observed"
    else:
        base, method = params.assumed_base_error_rate, "assumed"
    return {"method": method, "base_error_rate": base,
            "saturation_cpu": params.saturation_cpu, "assumption": True}


# ------------------------------------------------------------------ calibration

def calibrate(series: pd.DataFrame, params: CalibrationParams = CalibrationParams()
              ) -> tuple[dict, dict]:
    """Returns (models, fit): `models` goes into calibration.json; `fit` is the
    quality of the fit, with a chronological test (fit on the first part of the
    minutes, error measured on the rest)."""
    ok = series[(series["quality"] == "ok")
                & series["load_per_instance"].notna() & series["cpu"].notna()]
    ok = ok.sort_values("minute")
    if len(ok) < params.min_minutes:
        raise ValueError(f"Only {len(ok)} minutes with quality 'ok': at least "
                         f"{params.min_minutes} are needed to calibrate")
    split = ok["minute"].iloc[int(len(ok) * params.train_fraction)]
    lat = ok[ok["rt_ms"].notna() & (ok["cpu"] < 1)]

    cpu_block, cpu_r2 = fit_cpu(ok)
    lat_block = fit_latency(lat)

    # Chronological test: refit on the minutes before `split`, predict the rest.
    train, test = ok[ok["minute"] < split], ok[ok["minute"] >= split]
    cpu_train, _ = fit_cpu(train)
    cpu_test_error = mape(cpu_train["a"] * test["load_per_instance"].to_numpy()
                          + cpu_train["b"], test["cpu"].to_numpy())
    lat_train, lat_test = lat[lat["minute"] < split], lat[lat["minute"] >= split]
    rt_train = fit_latency(lat_train)
    rt_test_error = mape(rt_train["rt_base"] + rt_train["rt_queue"]
                         * queue_term(lat_test["cpu"].to_numpy()),
                         lat_test["rt_ms"].to_numpy())

    models = {"cpu_model": cpu_block, "latency_model": lat_block,
              "p95_model": fit_p95(lat, params), "error_model": fit_errors(ok, params)}
    fit = {"minutes": int(len(ok)), "cpu_r2": cpu_r2, "cpu_test_error": cpu_test_error,
           "latency_test_error": rt_test_error}
    return models, fit
