"""Q3: released PV forecast versions, paid contract adjustments, causal daily replay.

Settlement (main rule "refund"): the 0:00 plan q0 is billed at p. The last effective
contract q may deviate from it: the upward part costs 1.5p, the defaulted downward
part is billed at the 50 % default price, i.e. 0.5p of its original charge returns.
Rule "penalty" keeps the full plan charge and adds 0.5p per defaulted kWh. Every slot
is settled once, by comparing its last effective contract with the 0:00 plan.

Information chain (review round 1, items R02/R07). Release k of day j is issued at slot
b_k (issue_time) and covers the 144 slots that follow (valid_time = issue_time + lead,
lead = 1..144 slots): today's slots b_k..143 and tomorrow's slots 0..b_k-1. With
Policy.cross_day_release the advisory day j+1 uses that next-day coverage and fills
tomorrow's slots >= b_k with the in-house lead-2 forecast issued at 0:00 of day j; no
release of day j+1 is ever read. With Policy.matched_bias the calibration residuals
of the same-day horizon rebuild, for every completed historical day, the intraday
load-bias correction that release k of that day could itself have computed, so the
error library matches the predictor actually used. Both flags off reproduce the
frozen 2026-09-11 behaviour (legacy chain).
"""
from __future__ import annotations

import time
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


def tomorrow_pv_window(external, own, day, release, release_base, weight):
    """Combined PV forecast (kWh) for the 144 advisory slots of day+1 as seen at `release` of `day`.

    Tomorrow's slot s < release_base lies inside the release's 24 h coverage (index
    144-release_base+s of the version) and combines the release with the in-house lead-2
    forecast issued at 0:00 of `day`; slots s >= release_base lie beyond the coverage and
    keep the in-house forecast, the only source available at that time. The 0:00 release
    covers today only, so release_base == 0 returns the in-house forecast unchanged.
    """
    local = np.array(own[day, 1, :, 1], float)
    if release_base:
        local[:release_base] = weight*external[day, release, 144-release_base:]+(1-weight)*local[:release_base]
    return local


def tomorrow_scenarios(actual, own, external, day, release, release_base, weight, window):
    """Matured lead-one residual scenarios of tomorrow's net load with a release-aware PV point.

    The forecast history is built exactly like the current point: for every issue day i,
    the PV of day i+1 combines release `release` of day i (its next-day coverage) with the
    lead-2 in-house forecast of day i. A residual is used only when its target day i+1 is
    complete at `day`, i.e. i <= day-2; yesterday's forecast for today has not matured.
    With release_base == 0 this equals horizon_distributions(...)[1] of the 0:00 plan.
    """
    m = day-1
    if m < 1:
        raise ValueError("Insufficient completed history for the advisory day")
    pv = np.array(own[:m, 1, :, 1], float)
    if release_base:
        pv[:, :release_base] = weight*external[:m, release, 144-release_base:]+(1-weight)*pv[:, :release_base]
    history_forecast = np.stack([own[:m, 1, :, 0], pv], axis=-1)
    point = np.stack([own[day, 1, :, 0], tomorrow_pv_window(external, own, day, release, release_base, weight)], axis=-1)
    return net_distribution(actual[1:day], history_forecast, point, window), point


def sameday_scenarios(actual, own, external, day, release, release_base, start, weight, window, load_point,
                      history_bias=None):
    """Paired matured residual scenarios of net load for slots start..143 of `day`.

    The forecast history uses the same release index and combination for every
    completed day i <= day-1, so lead-matched errors enter; the PV point of `day`
    comes from the same release window, restricted to the slots at or after `start`.
    `history_bias[i]` (optional) is the intraday load-bias correction that day i could
    itself have applied at `start`; adding it makes the historical load predictor
    identical in form to the corrected point forecast (review item R07).
    """
    n = 144-start; lo = start-release_base
    if lo < 0 or n <= 0:
        raise ValueError("Release must not be later than the adjustment slot")
    history_actual = actual[:day, start:, :]
    local = own[:day, 0, start:, 1]
    history_load = own[:day, 0, start:, 0]
    if history_bias is not None:
        history_bias = np.asarray(history_bias, float)
        if history_bias.shape != (day, n):
            raise ValueError("Historical bias must cover every completed day and remaining slot")
        history_load = history_load+history_bias
    history_forecast = np.stack([history_load, weight*external[:day, release, lo:lo+n]+(1-weight)*local], axis=-1)
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


def historical_load_bias(actual, own, t, window, decay, days):
    """load_bias that slot t of every completed day i < days could itself have computed.

    Only actual[i, :t] and the 0:00 forecast of day i enter; every such day is complete
    at the time of use. Days without a defined 0:00 forecast yield NaN rows, which the
    residual library drops. Returns shape (days, 144-t).
    """
    n = 144-t
    if t == 0:
        return np.zeros((days, n))
    error = actual[:days, max(0, t-window):t, 0]-own[:days, 0, max(0, t-window):t, 0]
    return error.mean(axis=1)[:, None]*np.exp(-np.arange(n)/decay)[None, :]


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

    Proxy character (review item R05): the band end points follow from the price ratios
    in the single-slot, no-storage case, but the constant in-band unit value band_value
    is an additional piecewise-linear modelling choice, not implied by those end points.
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
    cross_day_release: bool = False # advisory day+1 uses the newest release's next-day coverage (R02)
    matched_bias: bool = False      # calibration residuals rebuild the intraday load-bias correction (R07)


def settlement_sign(settlement):
    if settlement == "refund":
        return -1.0
    if settlement == "penalty":
        return 1.0
    raise ValueError(f"Unknown settlement rule: {settlement}")


def run_day(day, actual, prices, own, external, release_slots, battery, dt, settings, pricing,
            policy, energy, plans=None, adjustments=None, price_model=None):
    """Causal replay of one day: 0:00 plan, paid adjustments at releases, greedy storage.

    Only rows of `actual` before the current slot enter any decision; the current
    slot's supply and demand are treated as measurable during the slot (Q2 convention).
    With `price_model` (Q4), planning uses its causal forecasts and every bill is
    settled at its realized prices; otherwise `prices` serve both purposes.
    Records (optional) keep the complete planning horizons: the 288-slot 0:00 plan and,
    for every release, the remaining committed slots plus the 144 advisory slots.
    """
    n = 144; window = settings["residual_window_days"]
    plan_prices = np.tile(prices, 2) if price_model is None else np.asarray(price_model.plan(day), float)
    settle = prices if price_model is None else np.asarray(price_model.settle(day), float)
    if len(plan_prices) != 2*n or len(settle) != n or np.min(plan_prices) <= 0:
        raise ValueError("Planning prices must cover 288 positive slots and settlement 144 slots")
    sign = settlement_sign(policy.settlement); up_mult = pricing["upward_adjustment_multiplier"]
    down_mult = pricing["downward_adjustment_penalty_multiplier"]; multiplier = pricing["emergency_multiplier"]
    load_point0 = own[day, 0, :, 0]
    clock = time.perf_counter(); solves = 1
    today, point0 = sameday_scenarios(actual, own, external, day, 0, 0, 0, policy.weight, window, load_point0)
    tomorrow = horizon_distributions(actual[:day], own[:day], own[day, :2], window)[1]
    risk_today = risk_profile(today, policy.plan_quantile); risk_tomorrow = risk_profile(tomorrow, policy.plan_quantile)
    point_tomorrow0 = np.array(own[day, 1, :, :], float)
    plan = plan_day(np.r_[risk_today, risk_tomorrow], plan_prices, energy, battery, dt, 0.0)
    q0, reference = commit_first_day(plan, n)
    contract = q0.copy(); version = np.zeros(n, int); latest_pv = point0[:, 1].copy()
    if plans is not None:
        plans.append({"day": day, "plan": q0.copy(), "risk": risk_today.copy(), "point": point0.copy(),
                      "energy": reference.copy(), "horizon_end_kwh": float(plan.energy[-1]), "objective": plan.objective,
                      "advisory_plan": plan.contract[n:].copy(), "advisory_risk": risk_tomorrow.copy(),
                      "advisory_point": point_tomorrow0, "advisory_energy": plan.energy[n:].copy(),
                      "charge": plan.charge.copy(), "discharge": plan.discharge.copy()})
    rows = []; max_residual = plan.max_residual; adjusted = []
    for t in range(n):
        if t in release_slots and release_slots.index(t) in policy.releases:
            k = release_slots.index(t)
            bias = load_bias(actual[day, :, 0], load_point0, t, settings["load_bias_window_slots"],
                             settings["load_bias_decay_slots"], n-t)
            release, base = (0, 0) if policy.freeze_forecast else (k, t)
            history_bias = None
            if policy.matched_bias:
                history_bias = historical_load_bias(actual, own, t, settings["load_bias_window_slots"],
                                                    settings["load_bias_decay_slots"], day)
            remaining, point = sameday_scenarios(actual, own, external, day, release, base, t, policy.weight,
                                                 window, load_point0[t:]+bias, history_bias)
            low = risk_profile(remaining, policy.low_quantile); high = risk_profile(remaining, policy.high_quantile)
            if policy.cross_day_release:
                next_scenarios, point_next = tomorrow_scenarios(actual, own, external, day, release, base, policy.weight, window)
                risk_next = risk_profile(next_scenarios, policy.plan_quantile)
                # Coverage records the release information actually used: with weight 0 the
                # advisory day never reads attachment 3, so the covered range is empty.
                coverage = base if policy.weight > 0 else 0
                next_count = int(len(next_scenarios))
            else:
                risk_next = risk_tomorrow; point_next = point_tomorrow0; coverage = 0; next_count = int(len(tomorrow))
            committed = np.r_[np.ones(n-t, bool), np.zeros(n, bool)]
            horizon_prices = np.r_[prices[t:], prices] if price_model is None else np.asarray(price_model.horizon(day, t), float)
            if len(horizon_prices) != 2*n-t or np.min(horizon_prices) <= 0:
                raise ValueError("Horizon prices must cover the remaining and advisory slots")
            result = adjust_contract(np.r_[q0[t:], np.zeros(n)], np.r_[low, risk_next], np.r_[high, risk_next],
                                     horizon_prices, energy, battery, dt, committed,
                                     up_mult*horizon_prices, sign*down_mult*horizon_prices, horizon_prices, 0.0)
            solves += 1
            contract[t:] = result.contract[:n-t]; version[t:] = t; latest_pv[t:] = point[:, 1]
            max_residual = max(max_residual, result.max_residual)
            adjusted.append({"release_slot": t, "release_index": release, "up_kwh": float(result.up[:n-t].sum()),
                             "down_kwh": float(result.down[:n-t].sum()), "objective": result.objective,
                             "planned_day_end_kwh": float(result.energy[n-t]), "scenario_count": int(len(remaining)),
                             "advisory_scenario_count": next_count, "load_bias_kwh": float(bias[0]),
                             "advisory_covered_slots": int(coverage),
                             "advisory_purchase_kwh": float(result.contract[n-t:].sum()),
                             "advisory_end_kwh": float(result.energy[-1]),
                             "advisory_risk_shift_kwh": float(np.max(np.abs(risk_next-risk_tomorrow)))})
            if adjustments is not None:
                adjustments.append({"day": day, "release_slot": t, "release_index": release, "advisory_covered_slots": int(coverage),
                                    "plan": np.r_[q0[t:], np.zeros(n)], "contract": result.contract.copy(),
                                    "low": np.r_[low, risk_next], "high": np.r_[high, risk_next],
                                    "point": np.vstack([point, point_next]), "charge": result.charge.copy(),
                                    "discharge": result.discharge.copy(), "energy": result.energy.copy(),
                                    "covered": result.covered.copy(), "committed": committed.copy(),
                                    "load_bias": np.r_[bias, np.zeros(n)], "prices": np.array(horizon_prices, float)})
        row = execute_step(actual[day, t, 0], actual[day, t, 1], contract[t], energy, energy, battery, dt, greedy=True)
        rows.append((energy, *row)); energy = row[-1]
    executed = np.asarray(rows)
    up = np.maximum(contract-q0, 0); down = np.maximum(q0-contract, 0)
    cash = float(settle@(q0+up_mult*up+sign*down_mult*down+multiplier*executed[:, 2]))
    return {"executed": executed, "plan": q0, "contract": contract, "version": version, "point": point0,
            "risk": risk_today, "latest_pv": latest_pv, "cash": cash, "energy": energy,
            "plan_prices": plan_prices[:n], "settle_prices": settle,
            "max_residual": max_residual, "adjusted": adjusted, "scenario_count": int(len(today)),
            "plan_objective": plan.objective, "planned_day_end_kwh": float(reference[-1]),
            "solve_seconds": time.perf_counter()-clock, "lp_solves": solves}
