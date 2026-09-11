"""Independent Q3 accounting: physics, contract versions, settlement and aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .problem2_evaluation import empirical_cvar, emergency_intervals  # noqa: F401 (re-exported)
from .problem3 import settlement_sign


def trajectory(date, actual, result, prices, pricing, policy, metadata):
    n = len(prices)
    names = ["soc_start_kwh", "actual_grid_kwh", "emergency_grid_kwh", "charge_kwh",
             "discharge_kwh", "pv_curtailed_kwh", "plan_unused_kwh", "soc_end_kwh"]
    frame = pd.DataFrame(result["executed"], columns=names)
    frame["date"] = str(pd.Timestamp(date).date()); frame["slot"] = np.arange(n)
    frame["start_minute"] = np.arange(n)*10; frame["end_minute"] = np.arange(1, n+1)*10
    frame["interval"] = [f"{t//6:02d}:{t%6*10:02d}-{(t+1)//6:02d}:{(t+1)%6*10:02d}" for t in range(n)]
    frame["load_kwh"] = actual[:, 0]; frame["pv_kwh"] = actual[:, 1]
    frame["forecast_load_kwh"] = result["point"][:, 0]; frame["forecast_pv_kwh"] = result["point"][:, 1]
    frame["latest_forecast_pv_kwh"] = result["latest_pv"]; frame["risk_net_kwh"] = result["risk"]
    plan = result["plan"]; contract = result["contract"]
    frame["planned_grid_kwh"] = plan; frame["adjusted_grid_kwh"] = contract
    frame["adjust_up_kwh"] = np.maximum(contract-plan, 0); frame["adjust_down_kwh"] = np.maximum(plan-contract, 0)
    frame["contract_version"] = result["version"]
    frame["price_yuan_per_kwh"] = prices
    sign = settlement_sign(policy.settlement)
    frame["planned_cost_yuan"] = prices*plan
    frame["adjustment_cost_yuan"] = prices*(pricing["upward_adjustment_multiplier"]*frame.adjust_up_kwh
                                            + sign*pricing["downward_adjustment_penalty_multiplier"]*frame.adjust_down_kwh)
    frame["emergency_cost_yuan"] = prices*pricing["emergency_multiplier"]*frame.emergency_grid_kwh
    frame["cost_yuan"] = frame.planned_cost_yuan+frame.adjustment_cost_yuan+frame.emergency_cost_yuan
    frame["settlement"] = policy.settlement
    frame["adjustment_slots"] = "|".join(str(s) for s in metadata.pop("adjustment_slots"))
    for key, value in metadata.items():
        frame[key] = value
    return frame


def audit(frame, battery, dt, pricing, tolerance=2e-5, initial=None):
    def a(name):
        return frame[name].to_numpy(float)
    h, e, c, d, z, w = [a(k) for k in ("actual_grid_kwh", "emergency_grid_kwh", "charge_kwh",
                                       "discharge_kwh", "pv_curtailed_kwh", "plan_unused_kwh")]
    q0, q, u, v = [a(k) for k in ("planned_grid_kwh", "adjusted_grid_kwh", "adjust_up_kwh", "adjust_down_kwh")]
    l, pv, E, F, p = [a(k) for k in ("load_kwh", "pv_kwh", "soc_start_kwh", "soc_end_kwh", "price_yuan_per_kwh")]
    version = frame.contract_version.to_numpy(int); slot = frame.slot.to_numpy(int)
    settlements = frame.settlement.unique()
    if len(settlements) != 1:
        raise ValueError("Mixed settlement rules in one trajectory")
    sign = settlement_sign(settlements[0])
    allowed = {0}
    for text in frame.adjustment_slots.unique():
        if pd.isna(text):
            continue                     # no release: CSV round trip yields NaN
        allowed |= {int(float(x)) for x in str(text).split("|") if x}
    mx = lambda x: float(np.max(np.abs(x))) if len(np.atleast_1d(x)) else 0.0
    checks = {"balance_kwh": mx(h+e+pv+d-l-c-z),
              "state_kwh": mx(F-E-battery.charge_efficiency*c+d/battery.discharge_efficiency),
              "continuity_kwh": mx(E[1:]-F[:-1]), "contract_identity_kwh": mx(q-h-w),
              "adjustment_identity_kwh": mx(q-q0-u+v), "adjustment_exclusive_kwh": mx(np.minimum(u, v)),
              "nonnegative_kwh": float(max(0, -min(np.min(x) for x in (q0, q, u, v, h, e, c, d, z, w)))),
              "bounds_kwh": float(max(0, battery.soc_min_kwh-min(E.min(), F.min()), max(E.max(), F.max())-battery.soc_max_kwh)),
              "charge_limit_kwh": float(max(0, c.max()-battery.max_charge_power_kw*dt)),
              "discharge_limit_kwh": float(max(0, d.max()-battery.max_discharge_power_kw*dt)),
              "mutual_exclusion_kwh": mx(np.minimum(c, d)),
              "emergency_charge_kwh": mx(np.minimum(e, c)), "emergency_curtail_kwh": mx(np.minimum(e, z)),
              "curtail_limit_kwh": float(max(0, np.max(z-pv))),
              "unadjusted_plan_slots_kwh": mx((u+v)[version == 0]),
              "version_not_after_slot": float(np.max(np.maximum(version-slot, 0))),
              "version_allowed": 0.0 if set(np.unique(version)) <= allowed else 1.0,
              "planned_cost_yuan": mx(a("planned_cost_yuan")-p*q0),
              "adjustment_cost_yuan": mx(a("adjustment_cost_yuan")-p*(pricing["upward_adjustment_multiplier"]*u
                                                                    + sign*pricing["downward_adjustment_penalty_multiplier"]*v)),
              "emergency_cost_yuan": mx(a("emergency_cost_yuan")-p*pricing["emergency_multiplier"]*e),
              "cash_yuan": mx(a("cost_yuan")-a("planned_cost_yuan")-a("adjustment_cost_yuan")-a("emergency_cost_yuan"))}
    if initial is not None:
        checks["initial_kwh"] = abs(float(E[0])-initial)
    numeric = frame.drop(columns=["adjustment_slots"], errors="ignore").select_dtypes(include="number")
    if not np.isfinite(numeric).all().all():
        raise ValueError("Non-finite Q3 output")
    if "training_last_date" in frame:
        if not (pd.to_datetime(frame.training_last_date) < pd.to_datetime(frame.date)).all():
            raise ValueError("Future data entered day-ahead training")
    if max(checks.values()) > tolerance:
        raise ValueError(f"Q3 audit failed: {checks}")
    return {"passed": True, "tolerance": tolerance, "rows": len(frame), "maxima": checks}


def daily_summary(frame):
    energy = ["planned_grid_kwh", "adjusted_grid_kwh", "adjust_up_kwh", "adjust_down_kwh", "actual_grid_kwh",
              "emergency_grid_kwh", "plan_unused_kwh", "pv_curtailed_kwh", "charge_kwh", "discharge_kwh", "load_kwh", "pv_kwh"]
    money = ["planned_cost_yuan", "adjustment_cost_yuan", "emergency_cost_yuan", "cost_yuan"]
    sums = frame.groupby("date")[energy+money].sum()
    sums["initial_kwh"] = frame.groupby("date").soc_start_kwh.first()
    sums["terminal_kwh"] = frame.groupby("date").soc_end_kwh.last()
    sums["emergency_slots"] = (frame.emergency_grid_kwh > 2e-5).groupby(frame.date).sum()
    sums["adjusted_slots"] = ((frame.adjust_up_kwh+frame.adjust_down_kwh) > 2e-5).groupby(frame.date).sum()
    sums["load_mae_kw"] = ((frame.load_kwh-frame.forecast_load_kwh).abs()*6).groupby(frame.date).mean()
    sums["pv_mae_kw"] = ((frame.pv_kwh-frame.forecast_pv_kwh).abs()*6).groupby(frame.date).mean()
    sums["latest_pv_mae_kw"] = ((frame.pv_kwh-frame.latest_forecast_pv_kwh).abs()*6).groupby(frame.date).mean()
    return sums.reset_index()


def summary(frame, terminal_value):
    daily = daily_summary(frame)
    keys = ["planned_grid_kwh", "adjusted_grid_kwh", "adjust_up_kwh", "adjust_down_kwh", "actual_grid_kwh",
            "emergency_grid_kwh", "plan_unused_kwh", "pv_curtailed_kwh",
            "planned_cost_yuan", "adjustment_cost_yuan", "emergency_cost_yuan", "cost_yuan"]
    result = {k: float(frame[k].sum()) for k in keys}
    p = frame.price_yuan_per_kwh
    result["upward_cost_yuan"] = float((p*frame.adjust_up_kwh).sum()*1.5)
    result["downward_cost_yuan"] = float(result["adjustment_cost_yuan"]-result["upward_cost_yuan"])
    initial = float(frame.soc_start_kwh.iloc[0]); terminal = float(frame.soc_end_kwh.iloc[-1])
    result.update(days=len(daily), initial_kwh=initial, terminal_kwh=terminal,
                  value_adjusted_cost_yuan=result["cost_yuan"]-terminal_value*(terminal-initial),
                  emergency_cost_cvar95_yuan=empirical_cvar(daily.emergency_cost_yuan),
                  emergency_slots=int((frame.emergency_grid_kwh > 2e-5).sum()),
                  emergency_days=int((daily.emergency_slots > 0).sum()),
                  adjusted_slots=int(daily.adjusted_slots.sum()),
                  adjusted_days=int((daily.adjusted_slots > 0).sum()),
                  load_mae_kw=float(abs(frame.load_kwh-frame.forecast_load_kwh).mean()*6),
                  pv_mae_kw=float(abs(frame.pv_kwh-frame.forecast_pv_kwh).mean()*6),
                  latest_pv_mae_kw=float(abs(frame.pv_kwh-frame.latest_forecast_pv_kwh).mean()*6),
                  risk_pointwise_coverage=float(((frame.load_kwh-frame.pv_kwh) <= frame.risk_net_kwh).mean()))
    return result


def audit_decisions(logs, releases):
    """Every daily decision record must have a strict past cutoff and the configured releases."""
    for row in logs:
        issue = pd.Timestamp(row["date"]); cutoff = pd.Timestamp(row["training_last_date"])
        if cutoff != issue-pd.Timedelta(days=1):
            raise ValueError("Decision history cutoff changed")
        if [a["release_slot"] for a in row["adjustments"]] != list(releases):
            raise ValueError("Adjustment releases differ from the policy")
        if row["committed_slots"] != 144 or row["advisory_slots"] != 144:
            raise ValueError("Incorrect planning/commitment horizon")
    return {"passed": True, "days": len(logs), "release_slots": list(releases)}


def audit_adjustments(records, frame, battery, dt, tolerance=2e-5):
    """Saved adjustment horizons must be feasible and must match the executed contracts."""
    if not np.isfinite(records.select_dtypes(include="number")).all().all():
        raise ValueError("Non-finite adjustment record")
    worst = 0.0; count = 0
    for (date, release), part in records.groupby(["date", "release_slot"], sort=True):
        part = part.sort_values("slot").reset_index(drop=True); count += 1
        if not np.array_equal(part.slot, np.arange(release, 144)):
            raise ValueError("Adjustment horizon must cover the remaining slots of the day")
        E = part.soc_start_kwh.to_numpy(); F = part.soc_end_kwh.to_numpy()
        c = part.charge_kwh.to_numpy(); d = part.discharge_kwh.to_numpy()
        q = part.contract_kwh.to_numpy(); q0 = part.plan_kwh.to_numpy(); w = part.covered_kwh.to_numpy()
        low = part.net_low_kwh.to_numpy(); high = part.net_high_kwh.to_numpy()
        errors = [np.max(abs(F-E-battery.charge_efficiency*c+d/battery.discharge_efficiency)), np.max(abs(E[1:]-F[:-1])),
                  max(0, battery.soc_min_kwh-min(E.min(), F.min()), max(E.max(), F.max())-battery.soc_max_kwh),
                  max(0, c.max()-battery.max_charge_power_kw*dt, d.max()-battery.max_discharge_power_kw*dt),
                  max(0, -min(c.min(), d.min(), q.min(), w.min())), np.max(np.minimum(c, d)),
                  max(0, np.max(w-np.maximum(high-low, 0))), max(0, -np.min(q+d-c-low-w))]
        realized = frame[(frame.date == date) & (frame.slot >= release)]
        if len(realized) != 144-release:
            raise ValueError("Missing execution slots")
        errors.append(abs(E[0]-realized.soc_start_kwh.iloc[0]))
        errors.append(np.max(abs(q0-realized.planned_grid_kwh.to_numpy())))
        # Later releases may overwrite; the version column tells which contract survived.
        kept = realized.contract_version.to_numpy() == release
        if kept.any():
            errors.append(np.max(abs(q[kept]-realized.adjusted_grid_kwh.to_numpy()[kept])))
        worst = max(worst, *errors)
    if worst > tolerance:
        raise ValueError(f"Adjustment record audit failed: {worst}")
    return {"passed": True, "records": count, "max_residual_kwh": float(worst)}
