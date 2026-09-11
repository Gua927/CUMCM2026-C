"""Deterministic Q1 MILP, LP certificate, and lexicographic optimal dispatch."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, hstack, lil_matrix, vstack

from .common import Battery, make_schedule


@dataclass
class Solution:
    schedule: object
    solver: dict
    certificate: dict
    duals: object


def build_model(data, battery: Battery, dt, terminal=None):
    battery.validate()
    n = len(data)
    if n == 0 or dt <= 0:
        raise ValueError("Empty horizon or non-positive time step")
    vals = data[["load_kwh", "pv_kwh", "price_yuan_per_kwh"]].to_numpy(float)
    if not np.isfinite(vals).all() or (vals[:, :2] < 0).any() or (vals[:, 2] <= 0).any():
        raise ValueError("Q1 requires finite nonnegative energy and positive prices")
    terminal = battery.initial_soc_kwh if terminal is None else terminal
    if not battery.soc_min_kwh <= terminal <= battery.soc_max_kwh:
        raise ValueError("Terminal energy outside bounds")
    g, c, d, z = [np.arange(i*n, (i+1)*n) for i in range(4)]
    E = np.arange(4*n, 5*n+1)
    u = np.arange(5*n+1, 6*n+1)
    size = 6*n+1
    obj = np.zeros(size)
    obj[g] = vals[:, 2]
    lo, hi = np.zeros(size), np.zeros(size)
    cc, dd = battery.max_charge_power_kw*dt, battery.max_discharge_power_kw*dt
    hi[g], hi[c], hi[d], hi[z], hi[u] = vals[:, 0]+cc, cc, dd, vals[:, 1], 1
    lo[E], hi[E] = battery.soc_min_kwh, battery.soc_max_kwh
    lo[E[0]] = hi[E[0]] = battery.initial_soc_kwh
    lo[E[-1]] = hi[E[-1]] = terminal
    eq = lil_matrix((2*n, size))
    for t in range(n):
        eq[t, [g[t], c[t], d[t], z[t]]] = [1, -1, 1, -1]
        eq[n+t, [E[t+1], E[t], c[t], d[t]]] = [1, -1, -battery.charge_efficiency,
                                                              1/battery.discharge_efficiency]
    rhs = np.r_[vals[:, 0]-vals[:, 1], np.zeros(n)]
    ub = lil_matrix((2*n, size))
    for t in range(n):
        ub[t, [c[t], u[t]]] = [1, -cc]
        ub[n+t, [d[t], u[t]]] = [1, dd]
    integrality = np.zeros(size, dtype=int)
    integrality[u] = 1
    return dict(n=n, size=size, obj=obj, lo=lo, hi=hi, eq=eq.tocsr(), rhs=rhs,
                ub=ub.tocsr(), b_ub=np.r_[np.zeros(n), np.full(n, dd)],
                integrality=integrality, g=g, c=c, d=d, z=z, E=E, u=u)


def _solve(m, objective=None, extra_rows=(), extra_upper=(), time_limit=120):
    A = vstack([m["eq"], m["ub"]] + [csr_matrix(x.reshape(1, -1)) for x in extra_rows]).tocsc()
    lb = np.r_[m["rhs"], np.full(len(m["b_ub"])+len(extra_rows), -np.inf)]
    ub = np.r_[m["rhs"], m["b_ub"], extra_upper]
    start = perf_counter()
    res = milp(m["obj"] if objective is None else objective,
               integrality=m["integrality"], bounds=Bounds(m["lo"], m["hi"]),
               constraints=LinearConstraint(A, lb, ub),
               options={"mip_rel_gap": 1e-10, "time_limit": time_limit, "presolve": True})
    elapsed = perf_counter()-start
    if not res.success or res.status != 0 or res.x is None:
        raise RuntimeError(f"MILP failed: status={res.status}, {res.message}")
    info = dict(status=int(res.status), message=res.message, objective=float(res.fun),
                dual_bound=float(res.mip_dual_bound), relative_gap=float(res.mip_gap),
                nodes=int(res.mip_node_count), seconds=elapsed)
    return res.x, info


def solve(data, battery, dt=1/6, *, polish=True, terminal=None, time_limit=120,
          cost_tolerance=1e-7, throughput_tolerance=1e-6):
    m = build_model(data, battery, dt, terminal)
    x, primary = _solve(m, time_limit=time_limit)
    stages = {"primary_cost": primary}
    # Never exchange a material amount of money for a secondary objective.
    cost_tol, throughput_tol = cost_tolerance, throughput_tolerance
    if cost_tol < 0 or throughput_tol < 0:
        raise ValueError("Lexicographic tolerances must be nonnegative")
    if polish:
        throughput = np.zeros(m["size"])
        throughput[m["c"]] = throughput[m["d"]] = 1
        x, second = _solve(m, throughput, [m["obj"]], [primary["objective"]+cost_tol], time_limit)
        stages["secondary_throughput"] = second
        # Cyclic total variation of grid energy. Auxiliary variables are continuous.
        n, size = m["n"], m["size"]
        ext = {**m, "size": size+n, "obj": np.r_[m["obj"], np.zeros(n)],
               "lo": np.r_[m["lo"], np.zeros(n)],
               "hi": np.r_[m["hi"], np.full(n, max(m["hi"][m["g"]]))],
               "integrality": np.r_[m["integrality"], np.zeros(n, dtype=int)],
               "eq": hstack([m["eq"], csr_matrix((2*n,n))]).tocsr()}
        tv = lil_matrix((2*n, size+n))
        for t in range(n):
            prev = (t-1) % n
            # += handles n=1 where current and predecessor coincide.
            tv[t, m["g"][t]] += 1
            tv[t, m["g"][prev]] -= 1
            tv[t, size+t] = -1
            tv[n+t, m["g"][t]] -= 1
            tv[n+t, m["g"][prev]] += 1
            tv[n+t, size+t] = -1
        ext["ub"] = vstack([hstack([m["ub"], csr_matrix((2*n,n))]), tv]).tocsr()
        ext["b_ub"] = np.r_[m["b_ub"], np.zeros(2*n)]
        smooth = np.r_[np.zeros(size), np.ones(n)]
        xx, third = _solve(ext, smooth, [ext["obj"], np.r_[throughput,np.zeros(n)]],
                           [primary["objective"]+cost_tol, second["objective"]+throughput_tol], time_limit)
        stages["tertiary_grid_variation"] = third
        # MIP feasibility tolerances can leave tiny negative flows. Re-optimize
        # continuous quantities with the proved integer mode fixed and tighter LP tolerances.
        fixed_lo,fixed_hi=ext["lo"].copy(),ext["hi"].copy()
        fixed_lo[m["u"]]=fixed_hi[m["u"]]=np.rint(xx[m["u"]])
        polish_ub=vstack([ext["ub"],csr_matrix(ext["obj"].reshape(1,-1)),
                         csr_matrix(np.r_[throughput,np.zeros(n)].reshape(1,-1))]).tocsr()
        start=perf_counter()
        clean=linprog(smooth,A_eq=ext["eq"],b_eq=ext["rhs"],A_ub=polish_ub,
                      b_ub=np.r_[ext["b_ub"],primary["objective"]+cost_tol,second["objective"]+throughput_tol],
                      bounds=list(zip(fixed_lo,fixed_hi)),method="highs-ds",
                      options={"dual_feasibility_tolerance":1e-9,"primal_feasibility_tolerance":1e-9})
        if not clean.success:raise RuntimeError(f"Fixed-mode precision polish failed: {clean.message}")
        x=clean.x[:size]
        stages["fixed_mode_precision_polish"]={"status":int(clean.status),"message":clean.message,
                                              "objective":float(clean.fun),"seconds":perf_counter()-start}

    lp = linprog(m["obj"], A_eq=m["eq"], b_eq=m["rhs"], A_ub=m["ub"], b_ub=m["b_ub"],
                 bounds=list(zip(m["lo"],m["hi"])), method="highs-ds",
                 options={"dual_feasibility_tolerance":1e-9,"primal_feasibility_tolerance":1e-9})
    if not lp.success:
        raise RuntimeError(f"LP certificate failed: {lp.message}")
    ye, yi, yl, yu = lp.eqlin.marginals, lp.ineqlin.marginals, lp.lower.marginals, lp.upper.marginals
    gradient = m["obj"] - m["eq"].T@ye - m["ub"].T@yi - yl - yu
    dual_value = float(m["rhs"]@ye+m["b_ub"]@yi+m["lo"]@yl+m["hi"]@yu)
    certificate = {
        "lp_status":int(lp.status), "lp_objective_yuan":float(lp.fun),
        "dual_objective_yuan":dual_value,
        "lp_primal_dual_gap_yuan":abs(float(lp.fun)-dual_value),
        "dual_stationarity_max":float(np.max(np.abs(gradient))),
        "dual_sign_violation_max":float(max(0,np.max(yi),np.max(-yl),np.max(yu))),
        "lp_balance_residual_max":float(np.max(np.abs(m["eq"]@lp.x-m["rhs"]))),
        "selected_minus_lp_yuan":float(m["obj"]@x-lp.fun),
        "selected_minus_milp_bound_yuan":float(m["obj"]@x-primary["dual_bound"]),
        "lp_simultaneous_charge_discharge_kwh":float(np.max(np.minimum(lp.x[m["c"]],lp.x[m["d"]]))),
        "cost_tie_tolerance_yuan":cost_tol,"throughput_tie_tolerance_kwh":throughput_tol,
        "equality_multipliers":ye.tolist(),"inequality_multipliers":yi.tolist(),
        "lower_bound_multipliers":yl.tolist(),"upper_bound_multipliers":yu.tolist(),
    }
    if max(certificate[k] for k in ("lp_primal_dual_gap_yuan","dual_stationarity_max",
                                   "dual_sign_violation_max")) > 1e-5:
        raise RuntimeError("LP dual certificate did not validate")
    out = make_schedule(data,battery,dt,x[m["g"]],x[m["c"]],x[m["d"]],x[m["z"]],x[m["E"]])
    duals = data[["slot","interval"]].copy()
    duals["lp_marginal_net_load_yuan_per_kwh"] = ye[:m["n"]]
    duals["lp_marginal_state_rhs_yuan_per_internal_kwh"] = ye[m["n"]:]
    return Solution(out,stages,certificate,duals)
