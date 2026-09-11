"""Q4: causal real-time price forecasts for planning, realized prices for settlement.

Attachment 4 prices are unknown in advance. Planning uses a day-ahead forecast built
from prices of completed days only; at each release the same-day forecast may be shifted
by the mean deviation observed so far. Every bill (plan, adjustment, emergency) is
settled at the realized price of the delivery slot. Modes "perfect" (today's prices
known at 0:00) and "fixed" (attachment 1 profile) are information ablations.
"""
from __future__ import annotations

import numpy as np

METHODS = ("combined", "weekday", "regression", "lag7", "fixed")


class PriceModel:
    def __init__(self, actual_prices, net_actual, net_forecast, fixed_profile, method="combined", mode="forecast",
                 weeks=2, regression_window=56, intraday="additive", floor=0.01, min_history=14):
        self.P = np.asarray(actual_prices, float); self.net = np.asarray(net_actual, float)
        self.net_forecast = np.asarray(net_forecast, float); self.fixed = np.asarray(fixed_profile, float)
        if method not in METHODS or mode not in ("forecast", "perfect", "fixed") or intraday not in (None, "additive"):
            raise ValueError("Unknown price model configuration")
        if self.P.ndim != 2 or self.P.shape[1] != 144 or self.net.shape != self.P.shape or self.net_forecast.shape[1:] != (2, 144):
            raise ValueError("Price model arrays must be [day, 144] and [day, 2, 144]")
        self.method = method; self.mode = mode; self.weeks = weeks; self.window = regression_window
        self.intraday = intraday; self.floor = floor; self.min_history = min_history

    # ---- causal day-ahead components: only days < day enter -------------------------------
    def weekday_profile(self, day, lead):
        target = day+lead-1
        rows = [target-7*k for k in range(1, self.weeks+1) if 0 <= target-7*k < day]
        if not rows:
            return self.P[:day].mean(0) if day > 0 else self.fixed.copy()
        return self.P[rows].mean(0)

    def regression_profile(self, day, lead):
        if day < self.min_history:
            return self.weekday_profile(day, lead)
        rows = np.arange(max(0, day-self.window), day)
        x = self.net[rows]; y = self.P[rows]; xm = x.mean(0); ym = y.mean(0)
        slope = ((x-xm)*(y-ym)).sum(0)/np.maximum(((x-xm)**2).sum(0), 1e-9)
        return np.maximum(ym+slope*(self.net_forecast[day, lead-1]-xm), self.floor)

    def day_ahead(self, day, lead):
        if self.method == "fixed":
            return self.fixed.copy()
        if self.method == "lag7":
            row = day+lead-1-7
            return self.P[row].copy() if 0 <= row < day else self.weekday_profile(day, lead)
        if self.method == "weekday":
            return self.weekday_profile(day, lead)
        if self.method == "regression":
            return self.regression_profile(day, lead)
        return 0.5*self.weekday_profile(day, lead)+0.5*self.regression_profile(day, lead)

    # ---- interfaces used by the replay ------------------------------------------------------
    def today(self, day):
        if self.mode == "perfect":
            return self.P[day].copy()
        if self.mode == "fixed":
            return self.fixed.copy()
        return np.maximum(self.day_ahead(day, 1), self.floor)

    def tomorrow(self, day):
        if self.mode == "fixed":
            return self.fixed.copy()
        return np.maximum(self.day_ahead(day, 2), self.floor)

    def plan(self, day):
        return np.r_[self.today(day), self.tomorrow(day)]

    def horizon(self, day, t):
        """Prices for slots t..143 of `day` plus the advisory day; only slots < t are observed."""
        today = self.today(day)
        if self.mode == "forecast" and self.intraday == "additive" and t > 0:
            today[t:] = np.maximum(today[t:]+float(np.mean(self.P[day, :t]-today[:t])), self.floor)
        return np.r_[today[t:], self.tomorrow(day)]

    def settle(self, day):
        return self.P[day].copy()


def price_errors(model, start, end, releases):
    """MAE (yuan/kWh) of the day-ahead forecast by lead and of the same-day horizon after releases."""
    P = model.P; out = {"lead1": [], "lead2": []}
    for t in releases:
        out[f"remaining_{t}"] = []; out[f"stale_{t}"] = []
    for day in range(start, end):
        out["lead1"].append(np.abs(model.today(day)-P[day]).mean())
        if day+1 < len(P):
            out["lead2"].append(np.abs(model.tomorrow(day)-P[day+1]).mean())
        for t in releases:
            fresh = model.horizon(day, t)[:144-t]; stale = model.today(day)[t:]
            out[f"remaining_{t}"].append(np.abs(fresh-P[day, t:]).mean()); out[f"stale_{t}"].append(np.abs(stale-P[day, t:]).mean())
    return {k: float(np.mean(v)) for k, v in out.items()}
