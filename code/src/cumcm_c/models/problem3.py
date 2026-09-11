"""Q3: released PV forecast versions, paid contract adjustments, causal daily replay.

Settlement (main rule "refund"): the 0:00 plan q0 is billed at p. The last effective
contract q may deviate from it: the upward part costs 1.5p, the defaulted downward
part is billed at the 50 % default price, i.e. 0.5p of its original charge returns.
Rule "penalty" keeps the full plan charge and adds 0.5p per defaulted kWh. Every slot
is settled once, by comparing its last effective contract with the 0:00 plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse import lil_matrix

from .forecasting import net_distribution, risk_profile, horizon_distributions
from .problem2 import checked_lp, plan_day, commit_first_day, execute_step

RELEASE_SLOTS = (0, 36, 72, 108)


def issued_pv_energy(hourly, actual_pv_kwh, release_slots, dt):
    """external[j, k, s]: PV kWh for the 144 slots after release k of day j.

    Forecast hour h anchors the slot starting h hours after the release (alignment
    tested against attachment 2). The slot completed just before the release is the
    only other anchor; it is observed at release time, so no future value is used.
    """
    hourly = np.asarray(hourly, float)
    days, releases, hours = hourly.shape
    if hours != 24 or releases != len(release_slots) or not np.isfinite(hourly).all():
        raise ValueError("Attachment 3 tensor must be finite [day, release, 24 hours]")
    flat = np.asarray(actual_pv_kwh, float).reshape(-1)
    grid = np.arange(144); anchors = np.r_[-1, 6*np.arange(1, 25)]
    out = np.empty((days, releases, 144))
    for j in range(days):
        for k, base in enumerate(release_slots):
            g = j*144+base
            previous = flat[g-1] if g > 0 else 0.0
            out[j, k] = np.maximum(np.interp(grid, anchors, np.r_[previous, hourly[j, k]*dt]), 0)
    return out


def pv_window(external, own, day, release, release_base, weight):
    """Combined PV forecast (kWh) for the 144 slots after a release of `day`.

    own[day, lead, slot, 1] are in-house forecasts issued at 0:00 of `day` (lead 0 =
    today, lead 1 = tomorrow); the external version is weighted by `weight`.
    """
    local = np.r_[own[day, 0, release_base:, 1], own[day, 1, :release_base, 1]]
    return weight*external[day, release]+(1-weight)*local


def sameday_scenarios(actual, own, external, day, release, release_base, start, weight, window, load_point):
    """Paired matured residual scenarios of net load for slots start..143 of `day`.

    The forecast history uses the same release index and combination for every
    completed day i <= day-1, so lead-matched errors enter; the PV point of `day`
    comes from the same release window, restricted to the slots at or after `start`.
    """
    n = 144-start; lo = start-release_base
    if lo < 0 or n <= 0:
        raise ValueError("Release must not be later than the adjustment slot")
    history_actual = actual[:day, start:, :]
    local = own[:day, 0, start:, 1]
    history_forecast = np.stack([own[:day, 0, start:, 0],
                                 weight*external[:day, release, lo:lo+n]+(1-weight)*local], axis=-1)
    point_pv = pv_window(external, own, day, release, release_base, weight)[lo:lo+n]
    point = np.stack([np.asarray(load_point, float), point_pv], axis=-1)
    if len(point) != n:
        raise ValueError("Load point must cover the remaining slots")
    return net_distribution(history_actual, history_forecast, point, window), point


def load_bias(actual_load, forecast_load, t, window, decay, n):
    """Mean recent load error decayed over the remaining horizon; only slots < t enter."""
    if t == 0:
        return np.zeros(n)
    error = actual_load[max(0, t-window):t]-forecast_load[max(0, t-window):t]
    return float(np.mean(error))*np.exp(-np.arange(n)/decay)


@lru_cache(maxsize=16)
def adjustment_matrices(n, eta_c, eta_d):
    # Variables: u(n) v(n) w(n) c(n) d(n) s(n) E(n+1).
    size = 6*n+n+1
    eq = lil_matrix((2*n, size))
    for t in range(n):
        # (q0+u-v)+d = n_low+w+c+s  ->  u-v-w-c+d-s = n_low-q0
        eq[t, [t, n+t, 2*n+t, 3*n+t, 4*n+t, 5*n+t]] = [1, -1, -1, -1, 1, -1]
        eq[n+t, [6*n+t, 6*n+t+1, 3*n+t, 4*n+t]] = [-1, 1, -eta_c, 1/eta_d]
    return eq.tocsr()


@dataclass
class Adjustment:
    contract: np.ndarray
    up: np.ndarray
    down: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    energy: np.ndarray
    covered: np.ndarray
    objective: float
    max_residual: float


def adjust_contract(base_contract, net_low, net_high, prices, initial, battery, dt, committed,
                    up_price, down_price, band_value, terminal_value=0.0):
    """Re-optimise the remaining horizon around the settled 0:00 plan.

    Committed slots keep the plan q0 unless moved: up at up_price per kWh, down at
    down_price (negative = refund). Covering demand between the low and high risk
    profiles is worth band_value per kWh, i.e. the newsvendor band is served only when
    it is cheaper than the plan price. Advisory slots buy freely at prices and cannot
    move down. The result is a planning proxy; settlement uses the executed contract.
    """
    n = len(prices); ec, ed = battery.charge_efficiency, battery.discharge_efficiency
    committed = np.asarray(committed, bool)
    q0 = np.where(committed, np.asarray(base_contract, float), 0.0)
    band = np.where(committed, np.maximum(np.asarray(net_high)-np.asarray(net_low), 0), 0.0)
    eq = adjustment_matrices(n, ec, ed)
    obj = np.zeros(eq.shape[1])
    obj[:n] = np.where(committed, up_price, prices)
    obj[n:2*n] = np.where(committed, down_price, 0.0)
    obj[2*n:3*n] = -np.where(committed, band_value, 0.0)
    obj[3*n:5*n] = 1e-8
    obj[-1] = -terminal_value
    bounds = [(0, None)]*n
    bounds += [(0, float(x)) for x in q0]
    bounds += [(0, float(b)) for b in band]
    bounds += [(0, battery.max_charge_power_kw*dt)]*n+[(0, battery.max_discharge_power_kw*dt)]*n
    bounds += [(0, None)]*n+[(battery.soc_min_kwh, battery.soc_max_kwh)]*(n+1)
    bounds[6*n] = (initial, initial)
    result, residual = checked_lp(obj, eq, np.r_[np.asarray(net_low)-q0, np.zeros(n)], bounds)
    x = result.x; energy = x[6*n:]; delta = np.diff(energy)
    charge = np.maximum(delta, 0)/ec; discharge = np.maximum(-delta, 0)*ed
    w = np.clip(x[2*n:3*n], 0, band)
    contract = np.maximum(q0+x[:n]-x[n:2*n], 0)
    if np.min(contract+discharge-charge-(np.asarray(net_low)+w)) < -2e-5:
        raise RuntimeError("Cycle removal broke the adjustment balance")
    up = np.maximum(contract-q0, 0); down = np.maximum(q0-contract, 0)
    return Adjustment(contract, up, down, charge, discharge, energy, w, float(result.fun), residual)


@dataclass(frozen=True)
class Policy:
    releases: tuple            # release indices (1..3) at which contracts may be adjusted
    weight: float              # external forecast weight in the PV point forecast
    settlement: str            # "refund" or "penalty"
    plan_quantile: float
    low_quantile: float
    high_quantile: float
    freeze_forecast: bool = False   # adjust with the 0:00 PV version only


def settlement_sign(settlement):
    if settlement == "refund":
        return -1.0
    if settlement == "penalty":
        return 1.0
    raise ValueError(f"Unknown settlement rule: {settlement}")


def run_day(day, actual, prices, own, external, release_slots, battery, dt, settings, pricing,
            policy, energy, plans=None, adjustments=None):
    """Causal replay of one day: 0:00 plan, paid adjustments at releases, greedy storage.

    Only rows of `actual` before the current slot enter any decision; the current
    slot's supply and demand are treated as measurable during the slot (Q2 convention).
    """
    n = 144; window = settings["residual_window_days"]
    sign = settlement_sign(policy.settlement); up_mult = pricing["upward_adjustment_multiplier"]
    down_mult = pricing["downward_adjustment_penalty_multiplier"]; multiplier = pricing["emergency_multiplier"]
    load_point0 = own[day, 0, :, 0]
    today, point0 = sameday_scenarios(actual, own, external, day, 0, 0, 0, policy.weight, window, load_point0)
    tomorrow = horizon_distributions(actual[:day], own[:day], own[day, :2], window)[1]
    risk_today = risk_profile(today, policy.plan_quantile); risk_tomorrow = risk_profile(tomorrow, policy.plan_quantile)
    plan = plan_day(np.r_[risk_today, risk_tomorrow], np.tile(prices, 2), energy, battery, dt, 0.0)
    q0, reference = commit_first_day(plan, n)
    contract = q0.copy(); version = np.zeros(n, int); latest_pv = point0[:, 1].copy()
    if plans is not None:
        plans.append({"day": day, "plan": q0.copy(), "risk": risk_today.copy(), "point": point0.copy(),
                      "energy": reference.copy(), "horizon_end_kwh": float(plan.energy[-1]), "objective": plan.objective})
    rows = []; max_residual = plan.max_residual; adjusted = []
    for t in range(n):
        if t in release_slots and release_slots.index(t) in policy.releases:
            k = release_slots.index(t)
            bias = load_bias(actual[day, :, 0], load_point0, t, settings["load_bias_window_slots"],
                             settings["load_bias_decay_slots"], n-t)
            release, base = (0, 0) if policy.freeze_forecast else (k, t)
            remaining, point = sameday_scenarios(actual, own, external, day, release, base, t, policy.weight,
                                                 window, load_point0[t:]+bias)
            low = risk_profile(remaining, policy.low_quantile); high = risk_profile(remaining, policy.high_quantile)
            committed = np.r_[np.ones(n-t, bool), np.zeros(n, bool)]
            horizon_prices = np.r_[prices[t:], prices]
            result = adjust_contract(np.r_[q0[t:], np.zeros(n)], np.r_[low, risk_tomorrow], np.r_[high, risk_tomorrow],
                                     horizon_prices, energy, battery, dt, committed,
                                     up_mult*horizon_prices, sign*down_mult*horizon_prices, horizon_prices, 0.0)
            contract[t:] = result.contract[:n-t]; version[t:] = t; latest_pv[t:] = point[:, 1]
            max_residual = max(max_residual, result.max_residual)
            adjusted.append({"release_slot": t, "up_kwh": float(result.up[:n-t].sum()), "down_kwh": float(result.down[:n-t].sum()),
                             "objective": result.objective, "planned_day_end_kwh": float(result.energy[n-t]),
                             "scenario_count": int(len(remaining)), "load_bias_kwh": float(bias[0])})
            if adjustments is not None:
                adjustments.append({"day": day, "release_slot": t, "plan": q0[t:].copy(), "contract": contract[t:].copy(),
                                    "low": low, "high": high, "point": point, "charge": result.charge[:n-t],
                                    "discharge": result.discharge[:n-t], "energy": result.energy[:n-t+1],
                                    "covered": result.covered[:n-t]})
        row = execute_step(actual[day, t, 0], actual[day, t, 1], contract[t], energy, energy, battery, dt, greedy=True)
        rows.append((energy, *row)); energy = row[-1]
    executed = np.asarray(rows)
    up = np.maximum(contract-q0, 0); down = np.maximum(q0-contract, 0)
    cash = float(prices@(q0+up_mult*up+sign*down_mult*down+multiplier*executed[:, 2]))
    return {"executed": executed, "plan": q0, "contract": contract, "version": version, "point": point0,
            "risk": risk_today, "latest_pv": latest_pv, "cash": cash, "energy": energy,
            "max_residual": max_residual, "adjusted": adjusted, "scenario_count": int(len(today)),
            "plan_objective": plan.objective, "planned_day_end_kwh": float(reference[-1])}
