"""Fixed day-ahead contracts, cost-calibrated risk profiles, causal storage control.

This is a reproducible approximate policy, NOT a globally optimal stochastic policy.
Daily planning LPs and fixed-contract control LPs have checked solver certificates.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

SOLVER_AUDIT={"solves":0,"max_primal_dual_gap_yuan":0.0,"max_stationarity":0.0,"failures":0}
# `failures` counts solves rejected by status, residual or dual certificate. Every rejection
# raises and aborts the run, so a complete archive necessarily reports 0; the counter makes
# that a measurement of the completed run rather than a constant.


@dataclass
class Plan:
    contract: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    energy: np.ndarray
    objective: float
    max_residual: float


@lru_cache(maxsize=32)
def matrices(n, eta_c, eta_d, planning):
    # Planning: q,c,d,z,E. Control: c,d,E. Surplus and deficit modes are disjoint.
    offset = 4 if planning else 2
    size = offset*n+n+1
    eq = lil_matrix(((2 if planning else 1)*n, size))
    c = np.arange((1 if planning else 0)*n, (2 if planning else 1)*n)
    d = c+n
    E = np.arange(offset*n, size)
    for t in range(n):
        if planning:
            eq[t, [t, c[t], d[t], 3*n+t]] = [1, -1, 1, -1]
        eq[(n if planning else 0)+t, [E[t], E[t+1], c[t], d[t]]] = [-1, 1, -eta_c, 1/eta_d]
    return eq.tocsr(), c, d, E


def checked_lp(obj, eq, rhs, bounds):
    result = linprog(obj, A_eq=eq, b_eq=rhs, bounds=bounds, method="highs",
                     options={"primal_feasibility_tolerance": 1e-8,
                              "dual_feasibility_tolerance": 1e-8})
    if result.status != 0 or not result.success:
        SOLVER_AUDIT["failures"]+=1
        raise RuntimeError(f"Q2 LP failed: {result.status}: {result.message}")
    residual = float(np.max(np.abs(eq@result.x-rhs)))
    if residual > 2e-5:
        SOLVER_AUDIT["failures"]+=1
        raise RuntimeError(f"Q2 LP residual {residual}")
    lo=np.array([0 if x[0] is None else x[0] for x in bounds],float)
    hi=np.array([np.inf if x[1] is None else x[1] for x in bounds],float)
    finite=np.isfinite(hi)
    dual=float(rhs@result.eqlin.marginals+lo@result.lower.marginals+hi[finite]@result.upper.marginals[finite])
    gap=abs(float(result.fun)-dual)
    stationarity=float(np.max(abs(obj-eq.T@result.eqlin.marginals-result.lower.marginals-result.upper.marginals)))
    if gap>2e-5 or stationarity>2e-5:
        SOLVER_AUDIT["failures"]+=1
        raise RuntimeError("Q2 LP dual certificate failed")
    SOLVER_AUDIT["solves"]+=1
    SOLVER_AUDIT["max_primal_dual_gap_yuan"]=max(SOLVER_AUDIT["max_primal_dual_gap_yuan"],gap)
    SOLVER_AUDIT["max_stationarity"]=max(SOLVER_AUDIT["max_stationarity"],stationarity)
    return result, residual


def plan_day(net, prices, initial, battery, dt, terminal_value=0.0):
    """Q1 physics on a risk-adjusted net-load trajectory, with terminal value.

    Positive prices + free spill permit removing simultaneous cycling without
    increasing cost while preserving the SOC increment. We explicitly remove it.
    """
    net, prices = np.asarray(net), np.asarray(prices)
    if len(net) != len(prices) or not np.isfinite(net).all() or np.min(prices) <= 0:
        raise ValueError("Finite net load and positive prices required")
    n = len(net); ec, ed = battery.charge_efficiency, battery.discharge_efficiency
    eq, c, d, E = matrices(n, ec, ed, True)
    obj = np.zeros(eq.shape[1]);obj[:n] = prices;obj[E[-1]] = -terminal_value
    bounds = [(0, None)]*n+[(0, battery.max_charge_power_kw*dt)]*n
    bounds += [(0, battery.max_discharge_power_kw*dt)]*n+[(0, None)]*n
    bounds += [(battery.soc_min_kwh, battery.soc_max_kwh)]*(n+1)
    bounds[E[0]] = (initial, initial)
    result, residual = checked_lp(obj, eq, np.r_[net, np.zeros(n)], bounds)
    energy = result.x[E]; delta = np.diff(energy)
    charge = np.maximum(delta, 0)/ec;discharge = np.maximum(-delta, 0)*ed
    contract = np.maximum(net+charge-discharge, 0)
    value = float(prices@contract-terminal_value*energy[-1])
    if value > result.fun+2e-5:
        raise RuntimeError("Cycle removal increased objective")
    return Plan(contract, charge, discharge, energy, value, residual)


def commit_first_day(plan,slots):
    """A forecast tail is advisory, never a second signed contract."""
    if not 0<slots<=len(plan.contract):raise ValueError("Invalid commitment horizon")
    return plan.contract[:slots].copy(),plan.energy[:slots+1].copy()


def control_targets(net, contract, prices, initial, battery, dt, multiplier, terminal_value):
    """Remaining-horizon LP: no contract changes, no emergency-funded charging."""
    n = len(net);ec, ed = battery.charge_efficiency, battery.discharge_efficiency
    eq, c, d, E = matrices(n, ec, ed, False)
    surplus = np.maximum(contract-net, 0);deficit = np.maximum(net-contract, 0)
    obj = np.zeros(eq.shape[1]);obj[d] = -multiplier*prices
    obj[c] = 1e-8;obj[E[-1]] = -terminal_value
    bounds = [(0, min(x, battery.max_charge_power_kw*dt)) for x in surplus]
    bounds += [(0, min(x, battery.max_discharge_power_kw*dt)) for x in deficit]
    bounds += [(battery.soc_min_kwh, battery.soc_max_kwh)]*(n+1)
    bounds[E[0]] = (initial, initial)
    result, residual = checked_lp(obj, eq, np.zeros(n), bounds)
    return result.x[E], residual


def execute_step(load, pv, contract, energy, target, battery, dt, greedy=False, idle=False):
    """Only current measurements and an already issued target enter this function."""
    ec, ed = battery.charge_efficiency, battery.discharge_efficiency
    surplus = contract+pv-load
    c=d=0.0
    if not idle:
        if surplus >= 0:
            c = min(surplus, battery.max_charge_power_kw*dt, max(0, battery.soc_max_kwh-energy)/ec)
        else:
            available = max(0, energy-battery.soc_min_kwh)*ed
            desired = available if greedy else max(0, energy-target)*ed
            d = min(-surplus, battery.max_discharge_power_kw*dt, available, desired)
    # PV has priority; leftover paid contracts and PV curtailment are distinct.
    required = load+c-d
    used_pv = min(pv, required)
    h = min(contract, max(required-pv, 0))
    emergency = max(required-pv-h, 0)
    next_energy = energy+ec*c-d/ed
    return (h, emergency, c, d, pv-used_pv, contract-h, next_energy)


def execute_day(actual, point, contract, prices, initial, battery, dt, settings,
                multiplier=5, terminal_value=0.0, controller="mpc", reference=None,
                advisory_point=None,advisory_contract=None,advisory_prices=None):
    n = len(prices);net = actual[:, 0]-actual[:, 1];pred = point[:, 0]-point[:, 1]
    full_point=point if advisory_point is None else advisory_point
    full_contract=contract if advisory_contract is None else advisory_contract
    full_prices=prices if advisory_prices is None else advisory_prices
    if not (len(full_point)==len(full_contract)==len(full_prices)):
        raise ValueError("Advisory horizon lengths differ")
    if not (np.allclose(full_point[:n],point) and np.allclose(full_contract[:n],contract)
            and np.allclose(full_prices[:n],prices)):
        raise ValueError("Advisory prefix must equal committed day")
    pred=full_point[:,0]-full_point[:,1]
    energy = initial;rows=[];target = np.full(len(pred)+1, initial);max_residual=0.0
    cadence=settings["controller_update_slots"]
    for t in range(n):
        if controller == "mpc" and t % cadence == 0:
            # At t: earlier intervals and the currently measured net load only.
            window=settings["online_bias_window_slots"]
            bias = float(np.mean(net[max(0,t-window):t]-pred[max(0,t-window):t])) if t else 0.0
            future = pred[t:].copy()+bias*np.exp(-np.arange(len(pred)-t)/settings["online_bias_decay_slots"])
            future[0] = net[t]
            targets,residual=control_targets(future,full_contract[t:],full_prices[t:],energy,battery,dt,
                                             multiplier,terminal_value)
            target[t:] = targets;max_residual=max(max_residual,residual)
        if controller == "reference":
            target = reference
        row=execute_step(actual[t,0],actual[t,1],contract[t],energy,target[t+1],battery,dt,
                         greedy=controller=="greedy",idle=controller=="idle")
        rows.append((energy, *row));energy=row[-1]
    return np.asarray(rows),max_residual
