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
# How the 90% intervals are estimated: also part of the data_version.
INTERVAL_METHOD = "least squares, Newey-West (Bartlett, Andrews AR(1) lags), 90%"
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
    """Least squares y = slope × x + intercept, with 90% intervals.

    x and y must be in time order. Consecutive minutes are correlated (a busy
    minute is followed by a busy minute), so the usual standard errors
    sqrt(diag(σ² (XᵀX)⁻¹)) would be too small and the intervals too narrow.
    The covariance of the coefficients is therefore estimated with Newey-West
    (HAC) standard errors: Bartlett weights over L lags, with L chosen from the
    lag-1 autocorrelation of the residuals (Andrews' rule). With independent
    residuals L is 0 and the result is the usual one (up to the n/(n-2) factor).

    Also returns the correlation between slope and intercept: they are
    estimated together, and a repeated run must draw them together.
    """
    X = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coef
    cov = newey_west_covariance(X, residuals)
    se = np.sqrt(np.diag(cov))
    total = ((y - y.mean()) ** 2).sum()
    correlation = cov[0, 1] / (se[0] * se[1]) if se[0] > 0 and se[1] > 0 else 0.0
    return {
        "slope": float(coef[0]),
        "slope_ci90": [float(coef[0] - Z90 * se[0]), float(coef[0] + Z90 * se[0])],
        "intercept": float(coef[1]),
        "intercept_ci90": [float(coef[1] - Z90 * se[1]), float(coef[1] + Z90 * se[1])],
        "correlation": float(np.clip(correlation, -1, 1)),
        "r2": float(1 - residuals @ residuals / total) if total > 0 else None,
    }


def newey_west_lags(residuals: np.ndarray) -> int:
    """Number of lags for the Bartlett kernel, from the lag-1 autocorrelation ρ
    of the residuals: L = ceil(1.1447 (α n)^(1/3)), α = 4ρ² / ((1−ρ)²(1+ρ)²)
    (Andrews, 1991), at most n/4. Negative ρ is treated as 0."""
    n = len(residuals)
    if n < 3 or np.allclose(residuals, 0):
        return 0
    rho = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
    rho = min(max(rho, 0.0), 0.97) if np.isfinite(rho) else 0.0
    alpha = 4 * rho ** 2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    return int(min(np.ceil(1.1447 * (alpha * n) ** (1 / 3)), n // 4))


def newey_west_covariance(X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Covariance of the least-squares coefficients robust to autocorrelation:
    (XᵀX)⁻¹ S (XᵀX)⁻¹ × n/(n−2), with S = Σ_l w_l Σ_t g_t g_{t−l}ᵀ (+ transpose),
    g_t = x_t u_t, w_l = 1 − l/(L+1)."""
    n = len(residuals)
    inverse = np.linalg.pinv(X.T @ X)
    g = X * residuals[:, None]
    S = g.T @ g
    lags = newey_west_lags(residuals)
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        shifted = g[lag:].T @ g[:-lag]
        S += weight * (shifted + shifted.T)
    return inverse @ S @ inverse * n / max(n - 2, 1)


def mape(predicted, observed) -> float:
    """Mean absolute percentage error, as a fraction (0.024 = 2.4%)."""
    return float(np.mean(np.abs(predicted - observed) / np.abs(observed)))


# ------------------------------------------------------------------- the models

def fit_cpu(ok: pd.DataFrame) -> tuple[dict, float]:
    lpi = ok["load_per_instance"].to_numpy()
    fit = fit_linear(lpi, ok["cpu"].to_numpy())
    block = {"a": fit["slope"], "a_ci90": fit["slope_ci90"],
             "b": fit["intercept"], "b_ci90": fit["intercept_ci90"],
             "a_b_correlation": fit["correlation"],
             "load_per_instance_range": [float(lpi.min()), float(lpi.max())]}
    return block, fit["r2"]


def fit_latency(lat: pd.DataFrame) -> dict:
    cpu = lat["cpu"].to_numpy()
    fit = fit_linear(queue_term(cpu), lat["rt_ms"].to_numpy())
    return {"rt_base": fit["intercept"], "rt_base_ci90": fit["intercept_ci90"],
            "rt_queue": fit["slope"], "rt_queue_ci90": fit["slope_ci90"],
            "rt_base_rt_queue_correlation": fit["correlation"],
            "observed_cpu_max": float(cpu.max())}


def fit_p95(lat: pd.DataFrame, params: CalibrationParams) -> dict:
    """Learned from rt_p95_ms when the source has enough of it, otherwise assumed."""
    if "rt_p95_ms" in lat and lat["rt_p95_ms"].notna().sum() >= params.min_minutes:
        d = lat[lat["rt_p95_ms"].notna()]
        fit = fit_linear(queue_term(d["cpu"].to_numpy()), d["rt_p95_ms"].to_numpy())
        return {"method": "observed",
                "p95_base": fit["intercept"], "p95_base_ci90": fit["intercept_ci90"],
                "p95_queue": fit["slope"], "p95_queue_ci90": fit["slope_ci90"],
                "p95_base_p95_queue_correlation": fit["correlation"],
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
