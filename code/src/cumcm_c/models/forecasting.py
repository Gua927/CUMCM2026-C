"""Causal daily profiles and out-of-sample residual calibration; no attachment 3."""
from __future__ import annotations

import numpy as np


def smooth(x, width=3):
    """Circular within-day smoothing is applied only to already observed days."""
    return sum(np.roll(x, k, axis=-1) for k in range(-width//2+1, width//2+1))/width


def predict(history, mode="trend", window=28):
    """history shape (completed days, 144, 2), units kWh. Never accepts future rows."""
    history = np.asarray(history, float)
    n = len(history)
    if n < 7 or not np.isfinite(history).all():
        raise ValueError("At least seven complete finite historical days required")
    if mode == "seasonal":
        return np.maximum(np.stack([history[-7, :, 0], history[-1, :, 1]], axis=-1), 0)
    if mode != "trend":
        raise ValueError(mode)
    # Weekday is learned by lag 7, without assuming which calendar days are weekends.
    lag = history[np.arange(n-7, max(-1, n-window-1), -7), :, 0]
    ages = np.arange(len(lag))*7+7
    if len(lag) >= 2:
        centered = ages-ages.mean()
        slope = np.sum(centered[:, None]*(lag-lag.mean(0)), axis=0)/np.sum(centered**2)
        load = lag.mean(0)-ages.mean()*slope
    else:
        load = lag[0]
    # Local linear PV profile captures seasonal drift; shrink the slope to reduce noise.
    pv_hist = history[-min(14, n):, :, 1]
    x = np.arange(len(pv_hist), dtype=float); center = x-x.mean()
    slope = np.sum(center[:, None]*pv_hist, axis=0)/np.sum(center**2)
    pv = pv_hist.mean(0)+0.75*(len(pv_hist)-x.mean())*slope
    load = 0.75*smooth(load, 3)+0.25*load
    pv = smooth(pv, 3)
    return np.maximum(np.stack([load, pv], axis=-1), 0)


def build_forecasts(actual, settings):
    """All forecasts stored before exposing each target day's observations."""
    result = {}
    for mode in ("seasonal", "trend"):
        forecasts = np.full_like(actual, np.nan)
        for day in range(7, len(actual)):
            forecasts[day] = predict(actual[:day], mode, settings["forecast_window_days"])
        result[mode] = forecasts
    return result


def net_distribution(actual_history, forecast_history, point, window=56, error_scale=1.0):
    """Paired complete-day residual vectors retain temporal/load-PV dependence."""
    good = np.isfinite(forecast_history).all(axis=(1, 2))
    residuals = (actual_history[good]-forecast_history[good])[-window:]
    if len(residuals) == 0:
        return (point[:, 0]-point[:, 1])[None, :]
    scenarios = np.maximum(point[None, :, :]+error_scale*residuals, 0)
    return scenarios[:, :, 0]-scenarios[:, :, 1]


def risk_profile(scenarios, quantile):
    if not 0 < quantile < 1:
        raise ValueError("Quantile must lie in (0,1)")
    return np.quantile(scenarios, quantile, axis=0)
