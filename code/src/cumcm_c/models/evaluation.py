"""Independent physical audit, transparent baselines, and Q1 aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import make_schedule


def audit(schedule, battery, dt, terminal=None, tolerance=2e-5):
    """Recompute from exported flows; do not use optimization matrices."""
    a = schedule
    required = ["load_kwh","pv_kwh","price_yuan_per_kwh","planned_grid_kwh",
                "actual_grid_kwh","charge_kwh","discharge_kwh","pv_curtailed_kwh",
                "soc_start_kwh","soc_end_kwh","cost_yuan","plan_unused_kwh","emergency_grid_kwh"]
    if not np.isfinite(a[required].to_numpy(float)).all():
        raise ValueError("Non-finite values in dispatch")
    terminal = battery.initial_soc_kwh if terminal is None else terminal
    balance = a.actual_grid_kwh+a.emergency_grid_kwh+a.pv_kwh+a.discharge_kwh-a.load_kwh-a.charge_kwh-a.pv_curtailed_kwh
    transition = a.soc_end_kwh-a.soc_start_kwh-battery.charge_efficiency*a.charge_kwh+a.discharge_kwh/battery.discharge_efficiency
    energy = np.r_[a.soc_start_kwh.to_numpy(),a.soc_end_kwh.iloc[-1]]
    cost = float(np.dot(a.price_yuan_per_kwh,a.planned_grid_kwh+5*a.emergency_grid_kwh))
    loss = float(((1-battery.charge_efficiency)*a.charge_kwh+
                  (1/battery.discharge_efficiency-1)*a.discharge_kwh).sum())
    total_rhs = float((a.load_kwh-a.pv_kwh+a.pv_curtailed_kwh).sum()+energy[-1]-energy[0]+loss)
    violations = {
        "balance_max_kwh":float(abs(balance).max()),
        "state_transition_max_kwh":float(abs(transition).max()),
        "state_continuity_max_kwh":float(np.max(np.abs(a.soc_start_kwh.to_numpy()[1:]-a.soc_end_kwh.to_numpy()[:-1]),initial=0)),
        "initial_error_kwh":abs(float(energy[0])-battery.initial_soc_kwh),
        "terminal_error_kwh":abs(float(energy[-1])-terminal),
        "soc_lower_violation_kwh":float(max(0,battery.soc_min_kwh-min(energy))),
        "soc_upper_violation_kwh":float(max(0,max(energy)-battery.soc_max_kwh)),
        "charge_limit_violation_kwh":float(max(0,a.charge_kwh.max()-battery.max_charge_power_kw*dt)),
        "discharge_limit_violation_kwh":float(max(0,a.discharge_kwh.max()-battery.max_discharge_power_kw*dt)),
        "simultaneous_charge_discharge_kwh":float(np.maximum(np.minimum(a.charge_kwh,a.discharge_kwh),0).max()),
        "curtailment_upper_violation_kwh":float(max(0,(a.pv_curtailed_kwh-a.pv_kwh).max())),
        "nonnegative_violation_kwh":float(max(0,-a[["planned_grid_kwh","actual_grid_kwh","charge_kwh","discharge_kwh","pv_curtailed_kwh","emergency_grid_kwh","plan_unused_kwh"]].min().min())),
        "contract_identity_error_kwh":float(abs(a.actual_grid_kwh+a.plan_unused_kwh-a.planned_grid_kwh).max()),
        "cost_recalculation_error_yuan":abs(cost-float(a.cost_yuan.sum())),
        "aggregate_energy_error_kwh":abs(float((a.actual_grid_kwh+a.emergency_grid_kwh).sum())-total_rhs),
    }
    report = {"passed":all(v<=tolerance for v in violations.values()),
              "absolute_tolerance":tolerance,"violations":violations}
    if not report["passed"]:
        raise ValueError(f"Independent physical audit failed: {report}")
    return report


def summarize(a, dt):
    return {
        "cost_yuan":float(a.cost_yuan.sum()),"grid_kwh":float(a.planned_grid_kwh.sum()),
        "load_kwh":float(a.load_kwh.sum()),"pv_kwh":float(a.pv_kwh.sum()),
        "charge_kwh":float(a.charge_kwh.sum()),"discharge_kwh":float(a.discharge_kwh.sum()),
        "curtailed_kwh":float(a.pv_curtailed_kwh.sum()),
        "pv_utilization":float(1-a.pv_curtailed_kwh.sum()/a.pv_kwh.sum()) if a.pv_kwh.sum()>0 else None,
        "loss_kwh":float(a.conversion_loss_kwh.sum()),
        "initial_kwh":float(a.soc_start_kwh.iloc[0]),"terminal_kwh":float(a.soc_end_kwh.iloc[-1]),
        "minimum_kwh":float(min(a.soc_start_kwh.min(),a.soc_end_kwh.min())),
        "maximum_kwh":float(max(a.soc_start_kwh.max(),a.soc_end_kwh.max())),
        "peak_grid_kw":float(a.planned_grid_kwh.max()/dt),
        "cyclic_grid_variation_kwh":float(np.abs(a.planned_grid_kwh.to_numpy()-np.roll(a.planned_grid_kwh.to_numpy(),1)).sum()),
    }


def no_storage(data,battery,dt):
    n=len(data)
    net=data.load_kwh.to_numpy()-data.pv_kwh.to_numpy()
    return make_schedule(data,battery,dt,np.maximum(net,0),np.zeros(n),np.zeros(n),
                         np.maximum(-net,0),np.full(n+1,battery.initial_soc_kwh))


def price_rule(data,battery,dt,quantiles=(0.25,0.75)):
    """Known-day 25/75% price rule, with explicit terminal reachability repair.

    Not tuned on optimal costs. Charge PV surplus first; charge at low prices;
    discharge at high prices. Projection maintains a reachable final energy.
    """
    n=len(data)
    load,pv,price=[data[x].to_numpy() for x in ("load_kwh","pv_kwh","price_yuan_per_kwh")]
    low,high=np.quantile(price,quantiles)
    cc,dd=battery.max_charge_power_kw*dt,battery.max_discharge_power_kw*dt
    ec,ed=battery.charge_efficiency,battery.discharge_efficiency
    g,c,d,z=[np.zeros(n) for _ in range(4)]
    E=np.zeros(n+1); E[0]=battery.initial_soc_kwh
    max_future_discharge=np.r_[np.cumsum(np.minimum(dd,load)[::-1])[::-1]/ed,0.0]
    for t in range(n):
        if pv[t]>load[t]: delta=min(cc,pv[t]-load[t])*ec
        elif price[t]<=low: delta=cc*ec
        elif price[t]>=high: delta=-min(dd,max(load[t]-pv[t],0))/ed
        else: delta=0.0
        lower=max(battery.soc_min_kwh,E[t]-min(dd,load[t])/ed,
                  battery.initial_soc_kwh-(n-t-1)*cc*ec)
        upper=min(battery.soc_max_kwh,E[t]+cc*ec,
                  battery.initial_soc_kwh+max_future_discharge[t+1])
        if lower>upper+1e-8: raise ValueError("Rule cannot reach terminal target")
        E[t+1]=np.clip(E[t]+delta,lower,upper)
        change=E[t+1]-E[t]
        c[t]=max(change,0)/ec; d[t]=max(-change,0)*ed
        g[t]=max(load[t]+c[t]-pv[t]-d[t],0)
        z[t]=g[t]+pv[t]+d[t]-load[t]-c[t]
    return make_schedule(data,battery,dt,g,c,d,z,E)


def reporting_tables(schedule):
    selected=schedule[schedule.start_minute.isin([600,720,840,960,1080,1200])]
    purchase=selected[["interval","planned_grid_kwh","cost_yuan"]].reset_index(drop=True)
    rows=[]
    for start in range(0,1440,240):
        part=schedule[(schedule.start_minute>=start)&(schedule.start_minute<start+240)]
        rows.append({"interval":f"{start//60:02d}:00-{(start+240)//60:02d}:00",
                     "charge_kwh":float(part.charge_kwh.sum()),"discharge_kwh":float(part.discharge_kwh.sum())})
    return purchase,pd.DataFrame(rows)
