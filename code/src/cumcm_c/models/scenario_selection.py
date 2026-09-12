"""Causal candidate selection using explicit simulated purchase and emergency bills.

This is finite-candidate policy search, not a multistage stochastic optimum.
Second-day contracts and costs are advisory scenario estimates only.
"""
import numpy as np

from .forecasting import risk_profile
from .problem2 import plan_day


def paired_horizon_scenarios(actual_history, issued_history, points, window=56):
    """Match contiguous target days from one origin; every target must be mature.

    issued_history[origin, lead] was computed before observing origin. The same
    origin indexes every lead, retaining cross-day residual dependence.
    """
    actual = np.asarray(actual_history, float)
    issued = np.asarray(issued_history, float)
    points = np.asarray(points, float)
    n, h = len(actual), len(points)
    if len(issued) != n or h < 1 or window < 1:
        raise ValueError("Mismatched history cutoff or invalid horizon/window")
    origins = np.arange(max(0, n-h+1))
    if len(origins):
        good = np.isfinite(issued[origins, :h]).all(axis=(1, 2, 3))
        origins = origins[good][-window:]
    if not len(origins):
        return points[None].reshape(1, -1, 2), origins
    targets = origins[:, None] + np.arange(h)[None, :]
    residuals = actual[targets] - issued[origins, :h]
    scenarios = np.maximum(points[None] + residuals, 0)
    return scenarios.reshape(len(origins), -1, 2), origins


def greedy_scenario_costs(scenarios, contracts, prices, initial, battery, dt,
                          multiplier=5.0, terminal_value=0.0):
    """Vectorize across scenarios/candidates; time remains strictly sequential.

    Only net supply in the current slot drives the buffer. No future scenario
    values enter a storage action. Returned cash excludes terminal bookkeeping.
    """
    scenarios = np.asarray(scenarios, float)
    contracts = np.asarray(contracts, float)
    prices = np.asarray(prices, float)
    if (scenarios.ndim != 3 or scenarios.shape[2] != 2 or contracts.ndim != 2
            or scenarios.shape[1] != len(prices) or contracts.shape[1] != len(prices)
            or not len(scenarios) or not len(contracts)
            or not all(np.isfinite(x).all() for x in (scenarios, contracts, prices))
            or np.min(scenarios) < 0 or np.min(contracts) < 0 or np.min(prices) <= 0):
        raise ValueError("Finite nonnegative scenarios/contracts and positive aligned prices required")
    energy = np.full((len(contracts), len(scenarios)), float(initial))
    emergency_cost = np.zeros_like(energy)
    emergency_energy = np.zeros_like(energy)
    for t, price in enumerate(prices):
        surplus = contracts[:, t, None] + scenarios[None, :, t, 1] - scenarios[None, :, t, 0]
        charge = np.minimum(np.maximum(surplus, 0), np.minimum(
            battery.max_charge_power_kw*dt, np.maximum(battery.soc_max_kwh-energy, 0)/battery.charge_efficiency))
        discharge = np.minimum(np.maximum(-surplus, 0), np.minimum(
            battery.max_discharge_power_kw*dt, np.maximum(energy-battery.soc_min_kwh, 0)*battery.discharge_efficiency))
        emergency = np.maximum(-surplus-discharge, 0)
        energy += battery.charge_efficiency*charge - discharge/battery.discharge_efficiency
        emergency_energy += emergency
        emergency_cost += multiplier*price*emergency
    planned_cost = contracts@prices
    cash = planned_cost[:, None]+emergency_cost
    return {"cash": cash, "planned_cost": planned_cost, "emergency_cost": emergency_cost,
            "emergency_energy": emergency_energy, "terminal_kwh": energy,
            "score": cash-terminal_value*(energy-initial)}


def choose_plan(marginals, joint_scenarios, prices, initial, battery, dt, quantiles,
                baseline_quantile, multiplier=5.0, far_terminal_value=0.0, tolerance=0.01):
    """Generate a small fixed candidate set and score using the execution rule.

    Prefer the baseline in a numerical tie, then the smaller quantile. Common
    residual-value accounting applies to every candidate, not its LP objective.
    """
    candidates = sorted(set(map(float, quantiles)) | {float(baseline_quantile)})
    risks = [np.concatenate([risk_profile(s, q) for s in marginals]) for q in candidates]
    plans = [plan_day(risk, prices, initial, battery, dt, far_terminal_value) for risk in risks]
    costs = greedy_scenario_costs(joint_scenarios, np.stack([p.contract for p in plans]),
                                  prices, initial, battery, dt, multiplier,
                                  float(np.min(prices))/battery.charge_efficiency)
    scores = costs["score"].mean(axis=1)
    tied = np.flatnonzero(scores <= scores.min()+tolerance)
    selected = min(tied, key=lambda i: (candidates[i] != baseline_quantile, candidates[i]))
    records = [{"quantile": q, "mean_cash_yuan": float(costs["cash"][i].mean()),
                "planned_cost_yuan": float(costs["planned_cost"][i]),
                "mean_emergency_cost_yuan": float(costs["emergency_cost"][i].mean()),
                "mean_terminal_kwh": float(costs["terminal_kwh"][i].mean()),
                "mean_score_yuan": float(scores[i]), "selected": i == selected}
               for i, q in enumerate(candidates)]
    return plans[selected], risks[selected], candidates[selected], records
