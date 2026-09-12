"""Causal daily profiles, a fitted direct regression forecaster and matured residuals.

Every forecaster receives only completed days. `predict` never sees the target day or
any later row, and the regression coefficients are re-estimated at each issue from the
most recent completed days only (rolling origin, direct multi-lead strategy).
"""
from __future__ import annotations

import numpy as np

FORECAST_MODES = ("seasonal", "trend", "regression")


def smooth(x, width=3):
    """Circular within-day smoothing is applied only to already observed days."""
    return sum(np.roll(x, k, axis=-1) for k in range(-width//2+1, width//2+1))/width


def _ridge_per_slot(X, y, ridge):
    """Per-slot ridge regression with an unpenalised intercept.

    X: (slots, samples, features), y: (slots, samples). Features are standardised inside
    the training window; the returned callable maps raw (slots, features) to predictions.
    """
    mean = X.mean(axis=1, keepdims=True); scale = X.std(axis=1, keepdims=True)
    scale = np.where(scale > 1e-9, scale, 1.0)
    Z = (X-mean)/scale; ybar = y.mean(axis=1, keepdims=True); yc = y-ybar
    k = Z.shape[2]; gram = np.einsum("smk,sml->skl", Z, Z)+ridge*np.eye(k)[None]
    beta = np.linalg.solve(gram, np.einsum("smk,sm->sk", Z, yc))
    return lambda x: ybar[:, 0]+np.einsum("sk,sk->s", (x-mean[:, 0, :])/scale[:, 0, :], beta)


def regression_predict(history, lead_days, training_days=56, ridge=1.0, min_samples=12):
    """Direct per-slot regression fitted on completed days only.

    Load at slot s of target day d is regressed on the same slot one week earlier, the
    same slot of the most recent completed day (d - lead) and that day's mean load; PV
    on the same slot of the most recent completed day and its seven-day mean. Training
    pairs (features, target) use targets d <= n-1, i.e. days whose actual profile is
    complete at the issue time; the issue-time feature row uses days <= n-1 as well.
    Returns None when fewer than `min_samples` training days exist.
    """
    history = np.asarray(history, float); n = len(history); lead = lead_days
    lo = max(7, lead+6, n-training_days); days = np.arange(lo, n)
    if len(days) < min_samples:
        return None
    load = history[:, :, 0]; pv = history[:, :, 1]; level = load.mean(axis=1)
    slots = load.shape[1]

    def load_features(d):
        d = np.atleast_1d(d)
        return np.stack([load[d-7], load[d-lead], np.repeat(level[d-lead][:, None], slots, axis=1)], axis=-1)

    def pv_features(d):
        d = np.atleast_1d(d)
        recent = np.stack([pv[d-lead-k] for k in range(7)], axis=0).mean(axis=0)
        return np.stack([pv[d-lead], recent], axis=-1)

    target = n+lead-1
    fit_load = _ridge_per_slot(load_features(days).transpose(1, 0, 2), load[days].T, ridge)
    fit_pv = _ridge_per_slot(pv_features(days).transpose(1, 0, 2), pv[days].T, ridge)
    out_load = fit_load(load_features(target)[0]); out_pv = fit_pv(pv_features(target)[0])
    return np.maximum(np.stack([smooth(out_load, 3), smooth(out_pv, 3)], axis=-1), 0)


def predict(history, mode="trend", window=28, lead_days=1, training_days=56, ridge=1.0):
    """history shape (completed days, 144, 2), units kWh. Never accepts future rows."""
    history = np.asarray(history, float)
    n = len(history)
    if not isinstance(lead_days,int) or not 1<=lead_days<=7:
        raise ValueError("Direct forecast lead must be 1..7 completed-day steps")
    if n < 7 or not np.isfinite(history).all():
        raise ValueError("At least seven complete finite historical days required")
    if mode == "seasonal":
        return np.maximum(np.stack([history[lead_days-8, :, 0], history[-1, :, 1]], axis=-1), 0)
    if mode == "regression":
        fitted = regression_predict(history, lead_days, training_days, ridge)
        # Too few completed days to estimate coefficients: fall back to the trend profile.
        return fitted if fitted is not None else predict(history, "trend", window, lead_days)
    if mode != "trend":
        raise ValueError(mode)
    # Weekday is learned by lag 7, without assuming which calendar days are weekends.
    target=n+lead_days-1
    indices=np.arange(target-7,max(-1,n-window-1),-7)
    lag = history[indices, :, 0]
    ages = target-indices
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
    pv = pv_hist.mean(0)+0.75*(len(pv_hist)+lead_days-1-x.mean())*slope
    load = 0.75*smooth(load, 3)+0.25*load
    pv = smooth(pv, 3)
    return np.maximum(np.stack([load, pv], axis=-1), 0)


def build_forecasts(actual, settings):
    """All forecasts stored before exposing each target day's observations."""
    result = {}
    for mode in FORECAST_MODES:
        forecasts = np.full_like(actual, np.nan)
        for day in range(7, len(actual)):
            forecasts[day] = predict(actual[:day], mode, settings["forecast_window_days"], 1,
                                     settings.get("regression_training_days", 56), settings.get("regression_ridge", 1.0))
        result[mode] = forecasts
    return result


def build_horizon_forecasts(actual,settings):
    """[issue day, lead index, slot, load/PV]; all leads share the same cutoff.

    Final issue day still predicts beyond Dec 31. No unknown actual row is needed.
    """
    h=settings["forecast_horizon_days"];result={}
    for mode in FORECAST_MODES:
        values=np.full((len(actual),h,*actual.shape[1:]),np.nan)
        for issue in range(7,len(actual)):
            for lead in range(h):
                values[issue,lead]=predict(actual[:issue],mode,settings["forecast_window_days"],lead+1,
                                           settings.get("regression_training_days",56),settings.get("regression_ridge",1.0))
        result[mode]=values
    return result


def horizon_distributions(actual_history,issued_history,points,window=56,error_scale=1.0):
    """Only matured, lead-matched forecast errors; target=origin+lead < now.

    Separate marginal profiles do not claim a calibrated joint multi-day law.
    """
    n=len(actual_history)
    if len(issued_history)!=n:raise ValueError("Forecast history and observation cutoff differ")
    output=[]
    for lead,point in enumerate(points):
        # At issue n, lead-2 forecasts from origin n-1 have NOT matured.
        if n<=lead:raise ValueError("Insufficient completed history")
        output.append(net_distribution(actual_history[lead:],issued_history[:n-lead,lead],
                                       point,window,error_scale))
    return output


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


def price_weighted_quantile(scenarios, prices, quantile):
    """Smallest value whose price-weighted cumulative mass reaches `quantile` (review item R01).

    With scenario prices P_s and net loads D_s, the single-slot cost E[P q + 5 P (D-q)^+]
    is minimised at the quantile of F^{(P)}(q) = E[P 1{D <= q}] / E[P], not of the plain
    demand distribution; when prices are constant across scenarios the two coincide.
    `scenarios` is (S, n) or (S,), `prices` broadcasts to it. This helper documents and
    tests the corrected condition; the published policies do not yet use it.
    """
    values = np.asarray(scenarios, float); single = values.ndim == 1
    values = values[:, None] if single else values
    weights = np.asarray(prices, float); weights = weights[:, None] if weights.ndim == 1 else weights
    weights = np.broadcast_to(weights, values.shape)
    if not 0 < quantile < 1 or np.any(weights < 0) or not np.isfinite(values).all():
        raise ValueError("Quantile in (0,1), non-negative prices and finite scenarios required")
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0); sorted_weights = np.take_along_axis(weights, order, axis=0)
    mass = np.cumsum(sorted_weights, axis=0)/np.maximum(sorted_weights.sum(axis=0, keepdims=True), 1e-300)
    index = np.argmax(mass >= quantile-1e-12, axis=0)
    out = np.take_along_axis(sorted_values, index[None, :], axis=0)[0]
    return float(out[0]) if single else out
